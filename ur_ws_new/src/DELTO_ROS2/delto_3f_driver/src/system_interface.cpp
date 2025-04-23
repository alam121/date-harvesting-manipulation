#include "system_interface.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace delto_3f_driver 
{
    
hardware_interface::return_type SystemInterface::configure(const hardware_interface::HardwareInfo & system_info)
{
    info_ = system_info;

    positions.resize(info_.joints.size(), 0.0);
    velocities.resize(info_.joints.size(), 0.0);
    effort_commands.resize(info_.joints.size(), 0.0);

    delto_ip = info_.hardware_parameters.at("delto_ip");
    delto_port = std::stoi(info_.hardware_parameters.at("delto_port"));

    float k_p = std::stod(info_.hardware_parameters.at("p_gain"));
    float k_d = std::stod(info_.hardware_parameters.at("d_gain"));

    p_gain.resize(12, k_p);
    d_gain.resize(12, k_d);

    for (const hardware_interface::ComponentInfo & joint : info_.joints) 
    {
        RCLCPP_ERROR(
            rclcpp::get_logger("SystemInterface"), "'%s' needs a %s command interface.",
            joint.name.c_str(), joint.command_interfaces[0].name.c_str());

        if (joint.command_interfaces[0].name != hardware_interface::HW_IF_EFFORT) 
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("SystemInterface"), "Joint '%s' needs an effort command interface.",
                joint.name.c_str());
            return hardware_interface::return_type::ERROR;
        }

        if (!(joint.state_interfaces[0].name == hardware_interface::HW_IF_POSITION ||
              joint.state_interfaces[1].name == hardware_interface::HW_IF_VELOCITY)) 
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("SystemInterface"),
                "Joint '%s' needs the following state interfaces in this order: %s, %s.",
                joint.name.c_str(), hardware_interface::HW_IF_POSITION, hardware_interface::HW_IF_VELOCITY);
            return hardware_interface::return_type::ERROR;
        }
    }

    delto_client = std::make_unique<DeltoTCP::Communication>(delto_ip, delto_port);
    m_init_thread = std::thread(&SystemInterface::init, this);
    m_init_thread.detach();

    return hardware_interface::return_type::OK;
}

hardware_interface::return_type SystemInterface::start()
{
    return hardware_interface::return_type::OK;
}

hardware_interface::return_type SystemInterface::stop()
{
    return hardware_interface::return_type::OK;
}

std::string SystemInterface::get_name() const
{
    return info_.name;
}

hardware_interface::status SystemInterface::get_status() const
{
    return hardware_interface::status::UNKNOWN;
}

std::vector<hardware_interface::StateInterface> SystemInterface::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> state_interfaces;
    state_interfaces.reserve(info_.joints.size() * 2);  // position, velocity

    std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t i = 0; i < info_.joints.size(); i++) 
    {
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_POSITION, &positions[i]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocities[i]));
    }

    return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> SystemInterface::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> command_interfaces;
    command_interfaces.reserve(info_.joints.size());

    for (size_t i = 0; i < info_.joints.size(); i++) 
    {
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
            info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &effort_commands[i]));
    }

    return command_interfaces;
}

hardware_interface::return_type SystemInterface::prepare_command_mode_switch(
  [[maybe_unused]] const std::vector<std::string> & start_interfaces,
  [[maybe_unused]] const std::vector<std::string> & stop_interfaces)
{
    return hardware_interface::return_type::OK;
}

void SystemInterface::init()
{
    delto_client->connect();
}

hardware_interface::return_type SystemInterface::read()
{
  try 
  {
    if (!delto_client) 
    {
        std::cerr << "Client is not initialized" << std::endl;
        return hardware_interface::return_type::ERROR;
    }

    auto received_data = delto_client->get_data();

    std::vector<double> new_positions;
    try 
    {
        if (received_data.joint.size() != 12)
        {
            std::cerr << "Received data size is not 12" << std::endl;
            return hardware_interface::return_type::ERROR;
        }
        
        new_positions = received_data.joint;
    }
    catch(const std::exception& e ) 
    {
        std::cerr << "Failed to read positions: " << e.what() << std::endl;
        return hardware_interface::return_type::ERROR;
    }

    if (new_positions.size() < positions.size()) 
    {
        std::cerr << "Insufficient position data. Expected: " << positions.size() 
                  << ", Got: " << new_positions.size() << std::endl;
        return hardware_interface::return_type::ERROR;
    }

    for (size_t i = 0; i < positions.size(); ++i) 
    {
        positions[i] = new_positions[i];
    }

    return hardware_interface::return_type::OK;
  }
  catch(const std::exception& e) 
  {
    std::cerr << "Unexpected error in read: " << e.what() << std::endl;
    return hardware_interface::return_type::ERROR;
  }
}

hardware_interface::return_type SystemInterface::write()
{
    std::vector<double> _duty = calc_duty(effort_commands);
    std::vector<int> int_duty(_duty.size());

    for (size_t i = 0; i < _duty.size(); ++i) 
    {
        int_duty[i] = static_cast<int>(_duty[i] * 10);
    }

    delto_client->send_duty(int_duty);

    return hardware_interface::return_type::OK;
}

std::vector<double> SystemInterface::calc_duty(std::vector<double> tq_u)
{
    std::vector<double> duty(12, 0.0);

    for (int i = 0; i < 12; ++i)
    {
        double v = 13.875 / 1.15 * tq_u[i];
        duty[i] = 100.0 * v / 11.1;

        duty[i] = std::max(-100.0, std::min(100.0, duty[i]));

    }

    return duty;
}

} // namespace delto_3f_driver

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(delto_3f_driver::SystemInterface, hardware_interface::SystemInterface)

