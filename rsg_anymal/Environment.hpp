// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

#include "AnymalController_20190837.hpp"
#include "AnymalController_opponent.hpp"

#define PLAYER2_NAME "opponent"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
    visualizable_(visualizable) {
    /// add objects
    ///BLUE IS SELF
    auto* robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robot->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    controller_.setOpponentName(PLAYER2_NAME);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// if you want make opponent robot, use like below code (but in Environment.hpp, there exist only PLAYER_NAME in definition. So use any name for opponent robot name setting.
    auto* dummy_robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    dummy_robot->setName(PLAYER2_NAME);
    dummyController_.setName(PLAYER2_NAME);
    dummyController_.setOpponentName(PLAYER_NAME);
    dummyController_.setPlayerNum(1);
    dummy_robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    dummyController_.create(&world_); //ADDED
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      server_->focusOn(robot);
      auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
      cage->setPosition(0,0,0);
    }
  }

  void init() {}

  void reset() {
    auto theta = uniDist_(gen_) * 2 * M_PI;
    auto offset = uniDist_(gen_) * (1.8* M_PI) - (0.9*M_PI);
    controller_.set_spawn_offset(offset);
    controller_.reset(&world_, theta);
    dummyController_.reset(&world_, theta); //ADDED
  }

  float step(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &opponentaction) {
    controller_.advance(&world_, action);
    dummyController_.advance(&world_, opponentaction); //ADDED
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservation(&world_);
    controller_.recordReward(&rewards_, &world_);
    dummyController_.updateObservation(&world_); //ADDED
    return rewards_.sum();
  }

  void observe(Eigen::Ref<EigenVec> ob, Eigen::Ref<EigenVec> opponentob) {
    controller_.updateObservation(&world_);
    dummyController_.updateObservation(&world_); //ADDED
    ob = controller_.getObservation().cast<float>();
    opponentob = dummyController_.getObservation().cast<float>();
  }

    ///ADDED
  bool contact() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
        if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
           contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
            return true;
        }
    }
    return false;
}

  bool opponent_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER2_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
        if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
           contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
            return true;
        }
    }
    /// get out of the cage
    int gcDim = anymal->getGeneralizedCoordinateDim();
    Eigen::VectorXd gc;
    gc.setZero(gcDim);
    gc = anymal->getGeneralizedCoordinate().e();
    if (gc.head(2).norm() > 3) {
        return true;
    }
    return false;
  }
  ///UP TO HERE

  bool isTerminalState(float &terminalReward) {
//    if(self_die()) {
//        terminalReward = terminalRewardCoeff_;
//        return true;
//    }
//    else if(opponent_die()){
//        terminalReward = 0;
//        return true;
//    }
//    return false;
//    if(contact()){
//        std::cout << "BASED TOUCHED" << std::endl;
//    }

    if(controller_.isTerminalState(&world_)) {
        terminalReward = terminalRewardCoeff_;
        std::cout << "Lose" << std::endl;
        return true;
    }
    if(dummyController_.isTerminalState(&world_)) {
        terminalReward = 1;
        std::cout << "Win" << std::endl;
        return true;
    }
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {
      if(rewards_.sum()>0.3){
          controller_.trained(true);
      }
      else{
          controller_.trained(false);
      }
  };

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return &world_; }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;
  AnymalController_20190837 controller_, dummyController_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}

