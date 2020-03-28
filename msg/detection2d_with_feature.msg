# This message contains a 2D bounding box corresponding to the detection of a person
# Also contains the feature of this person used for re-ID

Header header #header timestamp is time of frame acquisition


uint64 x1 # x coordinate of the top left of the bounding box
uint64 y1 # y coordinate of the top left of the bounding box
uint64 x2 # x coordinate of the bottom right of the bounding box
uint64 y2 # y coordinate of the bottom right of the bounding box

float64[] feature # re-ID feature

uint8 frame_det_id #unique id of this detection within this frame (used for associating 2D and 3D detections)
bool valid # whether detection is valid (within the boundaries of the image and has minimum required size)