/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

#include<queue>


namespace darknet_ros_depth {

char* cfg;
char* weights;
char* data;
char** detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh), imageTransport_(nodeHandle_), numClasses_(0), classLabels_(0), rosBoxes_(0), rosBoxCounter_(0), approx_sync_stereo_(0), depthTransport_(nodeHandle_) {
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector() {
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters() {
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_, std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init() {
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float)0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**)realloc((void*)detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_, 0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  // nodeHandle_.param("subscribers/depth_reading/topic", depthTopicName, std::string("/camera/stereo_recon/depth/image"));
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName, std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName, std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName, std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  // imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &YoloObjectDetector::cameraCallback, this);

  // ros::NodeHandle left_nh(nodeHandle_, "left");
  // image_transport::ImageTransport left_it(left_nh);
  approx_sync_stereo_ = new message_filters::Synchronizer<MyApproxSyncStereoPolicy>(
              MyApproxSyncStereoPolicy(500), imageSubscriber_, depthSubscriber_);  // larger queue resize may need (jwwoo)
  approx_sync_stereo_->registerCallback(
              boost::bind(&YoloObjectDetector::stereoCallback, this, _1, _2));


  // imageSubscriber_.subscribe(imageTransport_,"/kitti/camera_color_left/image_raw", 1);
  imageSubscriber_.subscribe(imageTransport_, cameraTopicName, 1);
  depthSubscriber_.subscribe(depthTransport_,"/camera/stereo_recon/depth/image", 1);


  // depthSubscriber_ = depthTransport_.subscribe("/kitti/stereo_recon/depth/image", cameraQueueSize, )
  objectPublisher_ =
      nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>(objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
  boundingBoxesPublisher_ =
      nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
  autowareDetectionPublisher_ = 
      nodeHandle_.advertise<autoware_msgs::DetectedObjectArray>("/autoware_detection/object_depth", 1, false);

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName, std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}


void YoloObjectDetector::stereoCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr &depth) {   
  ROS_DEBUG("[YoloObjectDetector] stereo image received.");

  cv_bridge::CvImagePtr cam_image;
  cv_bridge::CvImagePtr depth_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    depth_image = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image && depth_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
      camDepthCopy_ = depth_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
      // printf("imageStatus: %d\n", imageStatus_);
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB() {
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr = checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB() {
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const {
  return (ros::ok() && checkForObjectsActionServer_->isActive() && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage) {
  if (detectionImagePublisher_.getNumSubscribers() < 1) return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network* net) {
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network* net) {
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection* YoloObjectDetector::avgPredictions(network* net, int* nboxes) {
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for (j = 0; j < demoFrame_; ++j) {
    axpy_cpu(demoTotal_, 1. / demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  // detection* dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  detection* dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes, 1);
  return dets;
}

void* YoloObjectDetector::detectInThread() {
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float* X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float* prediction = network_predict(*net_, X);

  rememberNetwork(net_);
  detection* dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", fps_);
    printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_ + 2) % 3];
  image depth_map = buffd_[(buffIndex_ + 2) % 3];
  // draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_, 1);
  // draw_detections_vmy(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_, 1);


  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  // nboxes = nboxes < 3? nboxes : 3;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > 1) xmax = 1;
    if (ymax > 1) ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];

          
          // assert((int)x_center < depth_map.w && (int)y_center < depth_map.h);
          // printf("%d %d \n", depth_map.w, depth_map.h);
          // printf("%f \n", depth_map.data[(int)300*1242 + (int)300]);
          // printf("%d %d \n", (int)y_center, (int)x_center);
          // printf("%f %f \n", y_center, x_center);
          int real_y = y_center * frameWidth_;
          int real_x = x_center * frameHeight_;
          // printf("%d %d \n", real_x, real_y);
          int xmin_int = xmin * frameWidth_;
          int ymin_int = ymin * frameHeight_;
          int xmax_int = xmax * frameWidth_;
          int ymax_int = ymax * frameHeight_;
          // printf("%d %d %d %d\n", xmin_int, ymin_int, xmax_int, ymax_int);
          float median_depth = depth_map.data[real_y*depth_map.w + real_x];

          std::priority_queue<float> smaller; // Max heap
          std::priority_queue<float, std::vector<float>, std::greater<float> > bigger; // Min heap
          smaller.push(median_depth);
          for(int idx = xmin_int; idx <= xmax_int; idx++) {
            for(int idy = ymin_int; idy <= ymax_int; idy++) {
              // printf("%d %d\n", idx, idy);
              if( (idx < 0) || (idx >= depth_map.w) || (idy < 0) || (idy >= depth_map.h) ) continue;
              float number = (float)depth_map.data[idy*depth_map.w + idx];
              if(number <= 0.01) continue;
              // else debug_depth = number;
              if(smaller.empty()) {
                smaller.push(number);
                continue;
              }
              // printf("%d\n", number);
              if(number < smaller.top()){
                  // 기준보다 작은 경우 smaller에 넣는다.i
                  smaller.push(number);
              }
              else{
                  // 기준보다 큰 경우 bigger에 넣는다.
                  bigger.push(number);
              }
              // smaller의 크기와 bigger의 크기가 같거나, smaller의 크기가 하나 더 크게 유지되도록 데이터를 옮긴다. 
              if(smaller.size() < bigger.size()){
                  smaller.push(bigger.top());
                  bigger.pop();
              }
              else if(smaller.size() > bigger.size() + 1){
                  bigger.push(smaller.top());
                  smaller.pop();
              }
            }
          }
            // smaller와 bigger의 크기가 같다면 총 개수는 짝수이다.
          if(smaller.size() == bigger.size()){
              median_depth = ((float)smaller.top() + (float)bigger.top()) * 0.5f;
          }
          else{
              median_depth = (float)smaller.top();
          }

          // roiBoxes_[count].depth = depth_map.data[real_y*depth_map.w + real_x]; // core part (jwwoo)
          roiBoxes_[count].depth = median_depth;  // this is for publish
          dets[i].depth = median_depth; // this is for drawing
          // print(median_depth)
          count++;
        }
      }
    }
  }
  if(viewImage_)
    draw_detections_vmy2(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_, 1, 1);
  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void* YoloObjectDetector::fetchInThread() {
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
    IplImage* ROS_img = imageAndHeader.image;
    IplImage* ROS_depth = imageAndHeader.depth;
    ipl_into_image(ROS_img, buff_[buffIndex_]);
    headerBuff_[buffIndex_] = imageAndHeader.header;
    // printf("In fetch, headerBuff.seq: %d \n", headerBuff_[buffIndex_].seq);
    buffId_[buffIndex_] = actionId_;
    // ipl_into_image(ROS_depth, buffd_[buffIndex_]);

    float *data = (float *)ROS_depth->imageData;
    int h = ROS_depth->height;
    int w = ROS_depth->width;
    int c = ROS_depth->nChannels;
    int step = ROS_depth->widthStep / 4; // divide by four because its float, not int.
    // std::cout << h << ";" << w << ";" << c << ";" << step << ";\n";
    int i, j, k;
    // printf("%d %d %d", h, w, c);
    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                // std::cout << (float)buffd_[buffIndex_].data[k*w*h + i*w + j] << ", ";
                buffd_[buffIndex_].data[k*w*h + i*w + j] = (float)data[i*step + j*c + k] ;
            }
        }
    }
    // printf("ros: %f \n bufd: %f \n\n", (float)(data[300*step + 300*c]) , (buffd_[buffIndex_].data)[300*w + 300]);
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void* YoloObjectDetector::displayInThread(void* ptr) {

  #ifdef UNIQUE_FRAMES
  static int prevSeq;
  if (prevSeq==headerBuff_[(buffIndex_ + 1)%3].seq) {
      return 0;
  }
  prevSeq = headerBuff_[(buffIndex_ + 1)%3].seq;
  #endif

  show_image_cv(buff_[(buffIndex_ + 1) % 3], "YOLO V4");
  int c = cv::waitKey(waitKeyDelay_);
  if (c != -1) c = c % 256;
  if (c == 27) {
    demoDone_ = 1;
    return 0;
  } else if (c == 82) {
    demoThresh_ += .02;
  } else if (c == 84) {
    demoThresh_ -= .02;
    if (demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
    demoHier_ += .02;
  } else if (c == 81) {
    demoHier_ -= .02;
    if (demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void* YoloObjectDetector::displayLoop(void* ptr) {
  while (1) {
    displayInThread(0);
  }
}

void* YoloObjectDetector::detectLoop(void* ptr) {
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char* cfgfile, char* weightfile, char* datafile, float thresh, char** names, int classes, int delay,
                                      char* prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen) {
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image** alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V4\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo() {
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;
  std::thread publish_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float**)calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i) {
    predictions_[i] = (float*)calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float*)calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_*)calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
    IplImage* ROS_img = imageAndHeader.image;
    IplImage* ROS_depth = imageAndHeader.depth;
    buff_[0] = ipl_to_image(ROS_img);
    buffd_[0] = ipl_to_image(ROS_depth);
    headerBuff_[0] = imageAndHeader.header;
  }
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  buffd_[1] = copy_image(buffd_[0]);
  buffd_[2] = copy_image(buffd_[0]);
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);
  ipld_ = cvCreateImage(cvSize(buffd_[0].w, buffd_[0].h), IPL_DEPTH_32F, buffd_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cv::namedWindow("YOLO V4", cv::WINDOW_NORMAL);
    if (fullScreen_) {
      cv::setWindowProperty("YOLO V4", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } else {
      cv::moveWindow("YOLO V4", 0, 0);
      cv::resizeWindow("YOLO V4", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  while (!demoDone_) {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1. / (what_time_is_it_now() - demoTime_);
          demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      } else {
        generate_image(buff_[(buffIndex_ + 1) % 3], ipl_);
      }
      publishInThread();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }
}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader() {
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImage* ROS_depth = new IplImage(camDepthCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .depth = ROS_depth, .header = imageHeader_, };
  return header;
}

bool YoloObjectDetector::getImageStatus(void) {
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void) {
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void* YoloObjectDetector::publishInThread() {
  #ifdef UNIQUE_FRAMES
  static int prevSeq;
  if (prevSeq==headerBuff_[(buffIndex_ + 1)%3].seq) {
      return 0;
  }
  prevSeq = headerBuff_[(buffIndex_ + 1)%3].seq;
  #endif

  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "detection";
    msg.count = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;
        autoware_msgs::DetectedObject detectedObject;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.id = i;
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBox.depth = rosBoxes_[i][j].depth;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
          detectedObject.x = xmin;
          detectedObject.y = ymin;
          detectedObject.width = xmax - xmin;
          detectedObject.height = ymax - ymin;
          detectedObject.label = classLabels_[i];
          detectedObject.score = rosBoxes_[i][j].prob;
          detectedObject.id = i;
          // detectedObject.user_defined_info = rosBoxes_[i][j].depth;
          detectedObjectsResults_.objects.push_back(detectedObject);
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);

    detectedObjectsResults_.header.stamp = ros::Time::now();
    detectedObjectsResults_.header.frame_id = "detection";
    // detectedObjectsResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    autowareDetectionPublisher_.publish(detectedObjectsResults_);

  } else {
    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "detection";
    msg.count = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

} /* namespace darknet_ros*/
