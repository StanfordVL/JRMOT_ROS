# This message contains a 3D bounding box corresponding to the detection of a person
# Also contains the feature of this person used for re-ID

Header header #header timestamp is time of frame acquisition


float32 x # x coordinate of the center of the bottom face of the bounding box
float32 y # y coordinate of the center of the bottom face of the bounding box
float32 z # x coordinate of the center of the bottom face of the bounding box
float32 l # size of bounding box along x dimension
float32 h # size of bounding box along y dimension
float32 w # size of bounding box along z dimension
float32 theta # rotation of bounding box with respect to the positive x axis


float64[] feature # re-ID feature

uint8 frame_det_id #unique id of this detection within this frame (used for associating 2D and 3D detections)
bool valid # whether detection is valid (enough lidar points)