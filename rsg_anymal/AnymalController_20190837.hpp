// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_20190837 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 43; //default = 34
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  inline void set_spawn_offset(double theta_offset) {
      spawn_offset = theta_offset;      ///change relative spawn angle
  }

  inline bool reset(raisim::World *world, double theta) {
    if (playerNum_ == 0) {
      double distance = 1.5;
      if(trained_){
          distance = 2.5;
      }
      gc_init_.head(3) << distance * std::cos(theta + spawn_offset), distance * std::sin(theta + spawn_offset), 0.5;
      gc_init_.segment(3, 4) << cos((theta + spawn_offset - M_PI) / 2), 0, 0, sin((theta + spawn_offset - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 2.5 * std::cos(theta + M_PI), 2.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

  inline void updateObservation(raisim::World *world) {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// if you want use opponent robot`s state, use like below code
    auto opponent = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));
    Eigen::VectorXd opponentGc(gcDim_);
    Eigen::VectorXd opponentGv(gvDim_);
    opponent->getState(opponentGc, opponentGv);
    raisim::Vec<4> quat2;
    raisim::Mat<3, 3> rot2;
    quat2[0] = opponentGc[3];
    quat2[1] = opponentGc[4];
    quat2[2] = opponentGc[5];
    quat2[3] = opponentGc[6];
    raisim::quatToRotMat(quat2, rot2);
    opponentbodyLinearVel_ = rot2.e().transpose() * opponentGv.segment(0, 3);
    opponentbodyAngularVel_ = rot2.e().transpose() * opponentGv.segment(3, 3);
    ///////////////////////////////////////////////////////////////////////////////////////////////

      Eigen::Vector3d relative_velocity = rot.e().transpose() * (opponentGv.head(3) - gv_.head(3));  //vector to opponent from body frame
      Eigen::Vector3d to_opponent = rot.e().transpose() * (opponentGc.head(3) - gc_.head(3));  //vector to opponent from body frame
      Eigen::Vector3d relative_opponent_orientation = (rot.e().transpose() * rot2.e()).col(2);
      Eigen::Vector3d center = rot.e().transpose() * gc_.head(3);

      obDouble_ <<
        center, /// body pose (size 3, default size 1)
        rot.e().row(2).transpose(), /// direction of gravity vector in robot frame
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity

        ///Opponent's states:
        //relative_velocity, ///relative velocity of opponent in body frame (size 3)
        relative_opponent_orientation, ///(size 3)
        opponentGc[2],  ///ADDED
        to_opponent;     ///vector to opponent in body frame (size 3)

      ///size of obDouble: (34 + 2) + (1+3) = 40
  }

  inline void recordReward(Reward *rewards, raisim::World *world) {
      anymal_->getState(gc_, gv_);
      raisim::Vec<4> quat;
      raisim::Mat<3, 3> rot;
      quat[0] = gc_[3];
      quat[1] = gc_[4];
      quat[2] = gc_[5];
      quat[3] = gc_[6];
      raisim::quatToRotMat(quat, rot);
      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// if you want use opponent robot`s state, use like below code
      auto opponent = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));
      Eigen::VectorXd opponentGc(gcDim_);
      Eigen::VectorXd opponentGv(gvDim_);
      opponent->getState(opponentGc, opponentGv);
      raisim::Vec<4> quat2;
      raisim::Mat<3, 3> rot2;
      quat2[0] = opponentGc[3];
      quat2[1] = opponentGc[4];
      quat2[2] = opponentGc[5];
      quat2[3] = opponentGc[6];
      raisim::quatToRotMat(quat2, rot2);
      ///////////////////////////////////////////////////////////////////////////////////////////////

      ///std::cout << "rot col(2) =  " << rot(0,2) <<  ", " << rot(1,2) << ", " << rot(2,2) << std::endl;
//      if(!trained_){
//          rewards->record("torque", anymal_->getGeneralizedForce().squaredNorm());
//      }

      ///stay upright
      rewards->record("uprightness", abs(rot(2, 0)) + abs(rot(2, 1)));

      rewards->record("height", gc_[2]-0.4);

      Eigen::Vector2d to_opponent = opponentGc.head(2) - gc_.head(2);

      ///velocity towards opponent
      to_opponent.normalize();
      double projected_velocity = to_opponent.dot(gv_.head(2));
      rewards->record("velocity_to_opponent", projected_velocity);

      ///facing opponent (dot product)
      double facing = rot(0,0)*to_opponent(0) + rot(1,0)*to_opponent(1)/sqrt(rot.col(0).norm());
      rewards->record("facing", facing);

      ///push radially
      Eigen::Vector2d opponentvector = opponentGc.head(2);
      opponentvector.normalize();
      double radial = to_opponent.dot(opponentvector);
      rewards->record("radial", radial);

      ///insideness and contact
      double distance = gc_.head(2).norm();
      double opponent_distance = opponentGc.head(2).norm();

      if((opponent_distance - distance)>0){
          rewards->record("inside", 1);
      }
      else{
          rewards->record("outside", 1);
      }

      for (auto &contact: anymal_->getContacts()) {
          if(contact.getPairObjectIndex() == world->getObject("opponent")->getIndexInWorld() &&
             contact.getlocalBodyIndex() == anymal_->getBodyIdx("base")){
              rewards->record("pushing", 2*projected_velocity);
              if(trained_){
                  rewards->record("force", anymal_->getGeneralizedForce().squaredNorm());
              }
              ///if opponent is pushed off give reward
              if (opponentGc.head(2).norm() > 3) {
                  rewards->record("win", 1);
              }
          }
      }

//      ///Penalize radial velocity if more outside than opponent
//      Eigen::Vector2d radial = gc_.head(2);
//      radial.normalize();
//      double radial_velocity = radial.dot(gv_.head(2));
//      double insideness = opponent_distance - distance;
//      rewards->record("insideness", insideness);
//      if(insideness<0){
//          rewards->record("radial_velocity", radial_velocity);
//      }
  }

  void trained(bool status){
      trained_ = status;
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    for(auto& contact: anymal_->getContacts()) {
      if(contact.getPairObjectIndex() == world->getObject("ground")->getIndexInWorld() &&
         contact.getlocalBodyIndex() == anymal_->getBodyIdx("base")) {
          return true;
      }
    }
    if(gc_.head(2).norm() > 3) {
      return true;
    }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  Eigen::Vector3d opponentbodyLinearVel_, opponentbodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  double spawn_offset = 0;
  bool trained_ = false;
};

}