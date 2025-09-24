#pragma once

#include "toml.hpp"
#include <string>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>

const int FPS_RATE = []() {
    try {
        std::filesystem::path tomlCfg = "config.toml";
        std::filesystem::path configPath = ament_index_cpp::get_package_share_directory("assets") / tomlCfg;
        auto config = toml::parse_file(configPath.string());
        return config["ukf"]["fps_rate"].value_or(4);
    } catch (...) {
        return 4;
    }
}();