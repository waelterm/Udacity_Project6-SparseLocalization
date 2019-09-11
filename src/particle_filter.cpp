/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<> x_dist{ x,std[0] };
  std::normal_distribution<> y_dist{ y,std[1] };
  std::normal_distribution<> theta_dist{ theta,std[2] };


  for (int i = 0; i < num_particles; ++i)
  {
	  Particle particle;
	  particle.id = i;
	  particle.x = x_dist(gen);
	  particle.y = y_dist(gen);
	  particle.theta = theta_dist(gen);
	  particle.weight = 1.0;
	  particles.push_back(particle);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
	double velocity, double yaw_rate) {
	/**
	 * TODO: Add measurements to each particle and add random Gaussian noise.
	 * NOTE: When adding noise you may find std::normal_distribution
	 *   and std::default_random_engine useful.
	 *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	 *  http://www.cplusplus.com/reference/random/default_random_engine/
	 */
	std::default_random_engine gen;

	for (int i = 0; i < num_particles; ++i)
	{
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		double xf = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
		double yf = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		double thetaf = theta + yaw_rate * delta_t;

		std::normal_distribution<> x_dist{ xf, std_pos[0] };
		std::normal_distribution<> y_dist{ yf, std_pos[1] };
		std::normal_distribution<> theta_dist{ thetaf, std_pos[2] };

		particles[i].x = x_dist(gen);
		particles[i].y = y_dist(gen);
		particles[i].theta = theta_dist(gen);
	}
}

vector<double> ParticleFilter::vehicleToMap(double particle_x, double particle_y, double particle_theta, double obs_x, double obs_y) {
	vector<double> transformed;
	double x_m = particle_x + (cos(particle_theta) * obs_x) - (sin(particle_theta) * obs_y);
	double y_m = particle_y + (sin(particle_theta) * obs_x) + (cos(particle_theta) * obs_y);
	transformed.push_back(x_m);
	transformed.push_back(y_m);
	return transformed;
}

int ParticleFilter::dataAssociation(double obs_x, double obs_y, const Map& map_landmarks) {
	// for each observation:
	vector<double> distances;
	for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
	{
		double x_diff = (obs_x - map_landmarks.landmark_list[j].x_f);
		double y_diff = (obs_y - map_landmarks.landmark_list[j].y_f);
		double euclidean_distance = pow(x_diff * x_diff + y_diff * y_diff, 0.5);
		std::cout << "Euclid Distance " << j << ": " << euclidean_distance << std::endl;
		distances.push_back(euclidean_distance);
	}
	int minimum_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
	return minimum_index;

}

double ParticleFilter::multiVariateGaussian(double ox, double oy, double mu_x, double mu_y, double x, double y) {
	double p = 1 / (2 * M_PI * ox * oy) * exp(-(pow((x - mu_x), 2) / (2 * pow(ox, 2))) - pow(y - mu_y, 2) / (2 * pow(oy, 2)));
	return p;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	vector<double> weights;
	vector<int> associations;
	vector<double> sense_x;
	vector<double> sense_y;

	for (int i = 0; i < num_particles; ++i)
	{
		double weight = 1;
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;
		for (int i = 0; i < observations.size(); ++i) {
			double obs_x = observations[i].x;
			double obs_y = observations[i].y;
			vector<double> transformed = vehicleToMap(particle_x, particle_y, particle_theta, obs_x, obs_y);
			double tobs_x = transformed[0];
			double tobs_y = transformed[1];
			int minimum_index = dataAssociation(tobs_x, tobs_y, map_landmarks);
			associations.push_back(map_landmarks.landmark_list[minimum_index].id_i);
			sense_x.push_back(map_landmarks.landmark_list[minimum_index].x_f);
			sense_y.push_back(map_landmarks.landmark_list[minimum_index].y_f);
			double mu_x = tobs_x;
			double mu_y = tobs_y;
			double ox = std_landmark[0];
			double oy = std_landmark[1];
			double x = map_landmarks.landmark_list[minimum_index].x_f;
			double y = map_landmarks.landmark_list[minimum_index].y_f;
			double p = multiVariateGaussian(ox, oy, mu_x, mu_y, x, y);
			weight *= p;
		}
		particles[i].weight = weight;
		SetAssociations(particles[i], associations, sense_x, sense_y);
	}
	double weight_sum = accumulate(weights.begin(), weights.end(), 0);
	for (int i = 0; i < num_particles; ++i)
	{
		particles[i].weight /= weight_sum;
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    std::default_random_engine gen;
	vector<double> weights;
	for (int n = 0; n < num_particles; ++n) {
		weights.push_back(particles[n].weight);
	}

	std::discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> new_particles;
	for (int n = 0; n < num_particles; ++n) {
		new_particles.push_back(particles[d(gen)]);
	}
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}