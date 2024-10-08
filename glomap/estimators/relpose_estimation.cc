#include "glomap/estimators/relpose_estimation.h"

#include <PoseLib/robust.h>

#include <fstream> //修改
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glomap {

// 读取txt文件内容到unordered_map中，以image_id（行号）名作为key
std::unordered_map<image_t, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> ReadImagePosesFromTxt_id(const std::string& file_path) {
    std::unordered_map<image_t, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> image_poses;
    std::ifstream file(file_path);
    std::string line;

    image_t image_id = 1;  // 每一行对应一个image_id，顺序递增
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string image_name;
        double qw, qx, qy, qz, tx, ty, tz;

        // 跳过图片名，读取四元数和平移向量
        ss >> image_name >> qw >> qx >> qy >> qz >> tx >> ty >> tz;

        Eigen::Quaterniond quaternion(qw, qx, qy, qz); //wxyz
        Eigen::Vector3d translation(tx, ty, tz);

        image_poses[image_id] = {quaternion, translation};
        image_id++;  // 假设每一行的image_id递增
    }

    return image_poses;
}

void EstimateRelativePosesFromTxt_id(ViewGraph& view_graph,
                                  const std::string& txt_file_path) {
    // 从txt文件读取每个图像的姿态信息（四元数和平移向量）
    std::unordered_map<image_t, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> image_poses = ReadImagePosesFromTxt_id(txt_file_path);

    std::vector<image_pair_t> valid_pair_ids;
    for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
        if (!image_pair.is_valid) continue;
        valid_pair_ids.push_back(image_pair_id);
    }

    const int64_t num_image_pairs = valid_pair_ids.size();
    const int64_t kNumChunks = 10;
    const int64_t interval = std::ceil(num_image_pairs / kNumChunks);
    LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";

    for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
        std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%" << std::flush;
        const int64_t start = chunk_id * interval;
        const int64_t end = std::min<int64_t>((chunk_id + 1) * interval, num_image_pairs);

#pragma omp parallel for schedule(dynamic)
        for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
            ImagePair& image_pair = view_graph.image_pairs[valid_pair_ids[pair_idx]];

            // 从 image_poses 中获取两个图像的四元数和位移向量
            auto [quat1, trans1] = image_poses[image_pair.image_id1];
            auto [quat2, trans2] = image_poses[image_pair.image_id2]; //wxyz

            // 计算相对旋转矩阵 R2 * R1^T
            Eigen::Matrix3d R1 = quat1.toRotationMatrix();
            Eigen::Matrix3d R2 = quat2.toRotationMatrix(); //wxyz
            Eigen::Matrix3d R_rel = R2 * R1.transpose();

            // 将相对旋转矩阵转换回四元数
            Eigen::Quaterniond quat_rel(R_rel); //XYZW

            // 计算相对平移 T2 - T1
            Eigen::Vector3d trans_rel = trans2 - trans1;

            // 更新 image_pair.cam2_from_cam1 中的旋转和位移
            image_pair.cam2_from_cam1.rotation = quat_rel;
            image_pair.cam2_from_cam1.translation = trans_rel;

            // 输出结果进行验证
            std::cout << image_pair.image_id1 << std::endl;
            std::cout << image_pair.image_id2 << std::endl;
            std::cout << image_pair.cam2_from_cam1.rotation << std::endl;
        }
    }

    std::cout << "\r Estimating relative pose: 100%" << std::endl;
    LOG(INFO) << "Estimating relative pose done";
}

// 读取txt文件内容到unordered_map中，以图片名作为key
std::unordered_map<std::string, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> ReadImagePosesFromTxt_name(const std::string& file_path) {
    std::unordered_map<std::string, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> image_poses;
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string image_name;
        double qw, qx, qy, qz, tx, ty, tz;

        // 读取图片名，四元数和平移向量
        ss >> image_name >> qw >> qx >> qy >> qz >> tx >> ty >> tz;
        //std::cout<<qx<<" "<<qy<<" "<<qz<<" "<<qw<<std::endl;
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        Eigen::Vector3d translation(tx, ty, tz);

        // 将图片名作为key，存储四元数和位移向量
        image_poses[image_name] = {quaternion, translation};
        // std::cout<<quaternion<<std::endl;
    }

    return image_poses;
}

void EstimateRelativePosesFromTxt_name(ViewGraph& view_graph,
                                  std::unordered_map<image_t, Image>& images,
                                  const std::string& txt_file_path) {
    // 从txt文件读取每个图像的姿态信息（四元数和平移向量），以文件名作为键
    std::unordered_map<std::string, std::pair<Eigen::Quaterniond, Eigen::Vector3d>> image_poses = ReadImagePosesFromTxt_name(txt_file_path);

    std::vector<image_pair_t> valid_pair_ids;
    for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
        if (!image_pair.is_valid) continue;
        valid_pair_ids.push_back(image_pair_id);
    }

    const int64_t num_image_pairs = valid_pair_ids.size();
    const int64_t kNumChunks = 10;
    const int64_t interval = std::ceil(num_image_pairs / kNumChunks);
    LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";

    for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
        std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%" << std::flush;
        const int64_t start = chunk_id * interval;
        const int64_t end = std::min<int64_t>((chunk_id + 1) * interval, num_image_pairs);

#pragma omp parallel for schedule(dynamic)
        for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
            ImagePair& image_pair = view_graph.image_pairs[valid_pair_ids[pair_idx]];

            // 获取 image1 和 image2 的文件名
            const std::string& image_name1 = images[image_pair.image_id1].file_name;
            const std::string& image_name2 = images[image_pair.image_id2].file_name;
            // std::cout << "Image 1 Name: " << image_name1 << ", Image 2 Name: " << image_name2 << std::endl;

            // 从 image_poses 中获取两个图像的四元数和位移向量，使用文件名进行匹配
            if (image_poses.find(image_name1) == image_poses.end() || image_poses.find(image_name2) == image_poses.end()) {
                std::cerr << "Error: Could not find pose for image " << image_name1 << " or " << image_name2 << std::endl;
                continue;
            }

            auto [quat1_wxyz, trans1] = image_poses[image_name1]; //wxyz
            auto [quat2_wxyz, trans2] = image_poses[image_name2];

            // 计算相对旋转矩阵 R2 * R1^T
            Eigen::Matrix3d R1 = quat1_wxyz.toRotationMatrix();
            Eigen::Matrix3d R2 = quat2_wxyz.toRotationMatrix();
            Eigen::Matrix3d R_rel = R2.transpose() * R1;

            // 将相对旋转矩阵转换回四元数
            Eigen::Quaterniond quat_rel(R_rel); //wxyz

            // 计算相对平移 T2 - T1
            Eigen::Vector3d trans_rel = trans2 - trans1;

            // 更新 image_pair.cam2_from_cam1 中的旋转和位移
            // 手动调整四元数的存储顺序 wxyz → xyzw
            image_pair.cam2_from_cam1.rotation = quat_rel;
            // for (int i = 0; i < 4; ++i) {
            //     image_pair.cam2_from_cam1.rotation.coeffs()[i] = quat_rel.coeffs()[(i + 1) % 4];// xyzw
            // }
            image_pair.cam2_from_cam1.translation = trans_rel;

            // 输出结果进行验证
            // std::cout << image_pair.image_id1 << std::endl;
            // std::cout << image_pair.image_id2 << std::endl;
            // std::cout << image_pair.cam2_from_cam1.rotation << std::endl;
        }
    }

    std::cout << "\r Estimating relative pose: 100%" << std::endl;
    LOG(INFO) << "Estimating relative pose done";
}


void EstimateRelativePoses(ViewGraph& view_graph,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<image_t, Image>& images,
                           const RelativePoseEstimationOptions& options) {
  std::vector<image_pair_t> valid_pair_ids;
  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.push_back(image_pair_id);
  }

  // Define outside loop to reuse memory and avoid reallocation.
  std::vector<Eigen::Vector2d> points2D_1, points2D_2;
  std::vector<char> inliers;

  const int64_t num_image_pairs = valid_pair_ids.size();
  const int64_t kNumChunks = 10;
  const int64_t inverval = std::ceil(num_image_pairs / kNumChunks);
  LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";
  for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
    std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%"
              << std::flush;
    const int64_t start = chunk_id * inverval;
    const int64_t end =
        std::min<int64_t>((chunk_id + 1) * inverval, num_image_pairs);

#pragma omp parallel for schedule(dynamic) private( \
    points2D_1, points2D_2, inliers)
    for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
      ImagePair& image_pair = view_graph.image_pairs[valid_pair_ids[pair_idx]];
      const Image& image1 = images[image_pair.image_id1];
      const Image& image2 = images[image_pair.image_id2];
      const Eigen::MatrixXi& matches = image_pair.matches;

      // Collect the original 2D points
      points2D_1.clear();
      points2D_2.clear();
      for (size_t idx = 0; idx < matches.rows(); idx++) {
        points2D_1.push_back(image1.features[matches(idx, 0)]);
        points2D_2.push_back(image2.features[matches(idx, 1)]);
      }

      inliers.clear();
      poselib::CameraPose pose_rel_calc;

      

      poselib::estimate_relative_pose(
          points2D_1,
          points2D_2,
          ColmapCameraToPoseLibCamera(cameras[image1.camera_id]),
          ColmapCameraToPoseLibCamera(cameras[image2.camera_id]),
          options.ransac_options,
          options.bundle_options,
          &pose_rel_calc,
          &inliers);
      // Convert the relative pose to the glomap format
      for (int i = 0; i < 4; i++) {
        image_pair.cam2_from_cam1.rotation.coeffs()[i] =
            pose_rel_calc.q[(i + 1) % 4];// wxyz→xyzw
      }
      image_pair.cam2_from_cam1.translation = pose_rel_calc.t;
      std::cout << image_pair.image_id1 << std::endl;
      std::cout << image_pair.image_id2 << std::endl;
      std::cout << image_pair.cam2_from_cam1.rotation << std::endl;
    }
  }

  std::cout << "\r Estimating relative pose: 100%" << std::endl;
  LOG(INFO) << "Estimating relative pose done";
}

}  // namespace glomap
