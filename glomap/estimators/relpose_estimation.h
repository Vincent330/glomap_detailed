#pragma once

#include "glomap/scene/types_sfm.h"

#include <PoseLib/types.h>

namespace glomap {

struct RelativePoseEstimationOptions {
  // Options for poselib solver
  poselib::RansacOptions ransac_options;
  poselib::BundleOptions bundle_options;

  RelativePoseEstimationOptions() { ransac_options.max_iterations = 50000; }
};

void EstimateRelativePosesFromTxt_id(ViewGraph& view_graph,
                                  const std::string& txt_file_path);
void EstimateRelativePosesFromTxt_name(ViewGraph& view_graph,
                                  std::unordered_map<image_t, Image>& images,
                                  const std::string& txt_file_path);


void EstimateRelativePoses(ViewGraph& view_graph,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<image_t, Image>& images,
                           const RelativePoseEstimationOptions& options);

}  // namespace glomap
