#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <stdlib.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // random value generation
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution<> dist_x(x, std[0]);
  std::normal_distribution<> dist_y(y, std[1]);
  std::normal_distribution<> dist_theta(theta, std[2]);

  // set init value to all particles
  num_particles = 100;   // Set the number of particles.
  particles = std::vector<Particle>(num_particles);
  weights = std::vector<double>(num_particles);

  // Initialize all particles to first position (weights = 1.0)
  for(int i=0; i < particles.size(); i++){
    particles[i].x = dist_x(engine);
    particles[i].y = dist_y(engine);   
    particles[i].theta = dist_theta(engine);
    weights[i] = 1.0;    
  }

  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  
  // temp variables
  double stdx = std_pos[0];
  double stdy = std_pos[1];
  double stdth = std_pos[2];
  double v = velocity;
  double yr = yaw_rate;

  // random value generation
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution<> dist_x(0, stdx);
  std::normal_distribution<> dist_y(0, stdy);
  std::normal_distribution<> dist_theta(0, stdth);
  
  // Add measurements to each particle and add random Gaussian noise.
  for(int i=0; i < particles.size(); i++){
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double th0 = particles[i].theta;

    // model A : if yaw_rate == 0
    if(abs(yaw_rate) < 0.000001 ){
      particles[i].x = x0 + v * delta_t * cos(th0) + dist_x(engine);
      particles[i].y = y0 + v * delta_t * sin(th0) + dist_y(engine);
      particles[i].theta = th0 + dist_theta(engine);
    }
    
    // model B : if yaw_rate != 0
    else{
      particles[i].x = x0 + v/yr * ( sin(th0 + delta_t*yr) - sin(th0) ) + dist_x(engine);
      particles[i].y = y0 + v/yr * ( cos(th0) - cos(th0 + delta_t*yr) ) + dist_y(engine);
      particles[i].theta = th0 + delta_t*yr + dist_theta(engine);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  //  Find the predicted measurement that is closest to each observed measurement
  //  Assign the observed measurement to this particular landmark.
  
  for(int i=0; i < observations.size(); i++){
    LandmarkObs o = observations[i];
    double min_dist = numeric_limits<double>::max();   // DBL_MAX;
    double min_id = -1;
    
    for(int j=0; j < predicted.size(); j++){
      LandmarkObs p = predicted[j];

      // compute the distance
      double dist = sqrt( pow(o.x-p.x, 2) + pow(o.y-p.y, 2) );
      if(dist < min_dist){
	min_dist = dist;
	min_id = p.id;
      }
    }
    observations[i].id = min_id;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
				   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution.
  // The observations are given in the VEHICLE'S coordinate system.
  // The particles are located according to the MAP'S coordinate system.
  // Thus, it is necessary to transform between the two systems.
  
  for(int i=0; i < particles.size(); i++){

    // particle coordinate (vehicle coordinate)
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_th = particles[i].theta;    

    // find out the map_landmark_candidates [Map .x, .y (global coordinate)]
    vector<LandmarkObs> target_lm;

    // Q. How long does it take to retrieve the potential landmarks...?
    for(int j=0; j < map_landmarks.landmark_list.size(); j++){
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // find out the potential landmarks around the particle
      double dist_p_lm = sqrt( pow( (p_x-lm_x), 2) + pow( (p_y-lm_y), 2));
      if ( dist_p_lm <= sensor_range ){
	// add the landmarks in the sensor distance to target_lm
	LandmarkObs new_lm = {lm_id, lm_x, lm_y};
        target_lm.push_back(new_lm);
      }
    }

    // transform the original measurement value from vehicle coor. to global coordinate
    // 
    // <homogeneous transformation>
    // [cos(theta), -sin(theta), xt
    //  sin(theta),  cos(theta), yt
    //           0,           0, 1.0 ] * [x, y, 1]^T
    //
    vector<LandmarkObs> global_measurements;
    for(int j = 0; j < observations.size(); j++){
      LandmarkObs o = observations[j];
      double new_x = o.x * cos(p_th) - o.y * sin(p_th) + p_x;
      double new_y = o.x * sin(p_th) + o.y * cos(p_th) + p_y;

      // add global coorrinate measurement data
      LandmarkObs new_lm = {o.id, new_x, new_y};
      global_measurements.push_back(new_lm);
    }

    // data association with transformed measurement(global_measurements)
    // and map landmark(target_lm)
    dataAssociation(target_lm, global_measurements);
    
    // if the global_measurements.id(corresponding to the closest target_lm.id) is in the
    particles[i].weight = 1.0;
    for (int j = 0; j < global_measurements.size(); j++){
      LandmarkObs o = global_measurements[j];
      
      // find the corresponding ID
      LandmarkObs p;
      for (int k = 0; k < target_lm.size(); k++){
	p = target_lm[k];
	if (p.id == o.id){
	  // keep the p, which correpond to the closest landmarks
	  break;
	}
      }
      
      // weight by computing the maltivariate gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = 1.0 / sqrt(2 * M_PI * s_x * s_y) * exp( -1.0/2.0 * (pow((p.x - o.x), 2)/s_x + pow((p.y - o.y), 2)/s_y) );

      particles[i].weight *= obs_w;
      
    } // end of measurements
  }// end of particles
}

void ParticleFilter::resample() {
  
  vector<Particle> new_particles; // update the particles
  vector<double> weights; // current weights
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }  
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  std::random_device gen;
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  // resample wheel
  double beta = 0.0;
  int index = 0; // needs to be random?
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  // update the particle
  particles = new_particles;
  cout << particles[10].weight << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  // particle: the particle to assign each listed association,
  // and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
