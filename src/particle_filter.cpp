/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified on: Sep 10, 2019
 * Modified by: Philipp Waeltermann
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
	// Description: This function initializes the particle filter by initializing n particles 
	// with initial position and heading + noise  
	// x: Initial x-position from GPS  
	// y: Initial y-position from GPS
	// theta: Initial heading angle from GPS
	// std[]: Standard deviations of position give by GPS

	num_particles = 100;  //Setting the number of particles
	std::default_random_engine gen;
	std::normal_distribution<> x_dist{ x,std[0] };
	std::normal_distribution<> y_dist{ y,std[1] };
	std::normal_distribution<> theta_dist{ theta,std[2] };

	// Initializing particles
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
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
	double velocity, double yaw_rate) {
	// Description: This function updates the position of each particle by assuming a bicycle model. 
	// delta_t: Time passed in seconds since last set of measurements 
	// std_pos: Standard deviation of position information
	// velocity: Velocity of the vehicle
	// yaw_rate: Rate of change of the yaw angle of the vehicle

	std::default_random_engine gen;
	// Iterating through all particles
	for (int i = 0; i < num_particles; ++i)
	{
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		double xf;
		double yf;
		// Updating vehicle position according to bicycle model if yaw rate is not 0
		if (yaw_rate != 0)
		{
			xf = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			yf = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		}
		// Updating vehicle position according to constant velocity and yaw rate model if yaw rat is 0
		else
		{
			xf = x + velocity * delta_t * cos(theta);
			yf = y + velocity * delta_t * sin(theta);
		}
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
	// Description: This function uses the position and heading of the particle to convert an observation from vehicle coordinates to map coordinates. 
	// particle_x: x position associated with particle
	// particle_y: x position associated with particle
	// particle_theta: heading angle associated with particle
	// obs_x: x coordinate of observation in vehicle coordinates
	// obs_y: y coordinate of observation in vehicle coordinates
	// Returns: Vector of observation position in map coordinate system
	vector<double> transformed;
	double x_m = particle_x + (cos(particle_theta) * obs_x) - (sin(particle_theta) * obs_y);
	double y_m = particle_y + (sin(particle_theta) * obs_x) + (cos(particle_theta) * obs_y);
	transformed.push_back(x_m);
	transformed.push_back(y_m);
	return transformed;
}

int ParticleFilter::dataAssociation(double obs_x, double obs_y, const Map& map_landmarks, double range) {
	// Description: This function finds the closest landmark to an observation and returns its index 
	// obs_x: x coordinate of observation in map coordinates
	// obs_y: y coordinate of observation in map coordinates
	// map_landmarks: map including list of landmarks
	// range: range of sensors in meters
	// Returns: Index of closest landmark to observation

	vector<double> distances;
	// Iterate through landmarks
	for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
	{
		// Calculate euclidean distance
		double x_diff = (obs_x - map_landmarks.landmark_list[j].x_f);
		double y_diff = (obs_y - map_landmarks.landmark_list[j].y_f);
		double euclidean_distance = pow(x_diff * x_diff + y_diff * y_diff, 0.5);
		if (range > euclidean_distance)
		{
			distances.push_back(euclidean_distance);
		}
		else
		{
			distances.push_back(range);
		}
	}
	// Find and return index of landmark with smallest distance
	int minimum_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
	return minimum_index;

}

double ParticleFilter::multiVariateGaussian(double ox, double oy, double mu_x, double mu_y, double x, double y) {
	// Description: Calculates the Multi-Variate Gaussian of an observation and its associated landmark
	// ox: Standard deviation of x position of observation
	// oy: Standard deviation of y position of observation
	// mu_x: X position of observation
	// mu_y: Y position of observation
	// x: x position of associated landmark
	// y: y position of associated landmark
	// Returns: Multi-variate Gaussian
	double p = 1 / (2 * M_PI * ox * oy) * exp(-(pow((x - mu_x), 2) / (2 * pow(ox, 2))) - pow(y - mu_y, 2) / (2 * pow(oy, 2)));
	return p;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const vector<LandmarkObs>& observations,
	const Map& map_landmarks) {
	// Description: This function iterates through the particles and updates the particle weight based on the correlation of observations and landmarks on map
	// sensor_range: maximum range to detect landmark
	// std_landmark[]: Expected Standard deviation of observations
	// observations: List of observed landmarks in vehicle coordinates
	// map_landmarks: map including list of landmarks

	vector<double> weights; // List of weights of all particles
	// Iterating throug particles
	for (int i = 0; i < num_particles; ++i)
	{
		double weight = 1;
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		// Iterating through observations
		for (int i = 0; i < observations.size(); ++i)
		{
			double obs_x = observations[i].x;
			double obs_y = observations[i].y;
			// Transforming from vehicle coordinates to map coordinates
			vector<double> transformed = vehicleToMap(particle_x, particle_y, particle_theta, obs_x, obs_y);
			double tobs_x = transformed[0];
			double tobs_y = transformed[1];
			// Finding associated landmark
			int minimum_index = dataAssociation(tobs_x, tobs_y, map_landmarks, sensor_range);
			associations.push_back(map_landmarks.landmark_list[minimum_index].id_i);
			sense_x.push_back(map_landmarks.landmark_list[minimum_index].x_f);
			sense_y.push_back(map_landmarks.landmark_list[minimum_index].y_f);
			// Calculating multi-variate Gaussian for each observation
			double mu_x = tobs_x;
			double mu_y = tobs_y;
			double ox = std_landmark[0];
			double oy = std_landmark[1];
			double x = map_landmarks.landmark_list[minimum_index].x_f;
			double y = map_landmarks.landmark_list[minimum_index].y_f;
			double p = multiVariateGaussian(ox, oy, mu_x, mu_y, x, y);
			// The weight is the product of all the multi-variate Gaussian values for each observation
			weight *= p;
		}
		// Update weight of the particle
		particles[i].weight = weight;
		weights.push_back(weight);

		// Update asoociations ans sensing vectors of particle
		SetAssociations(particles[i], associations, sense_x, sense_y);
	}

	// Normalizing weights
	double weight_sum = accumulate(weights.begin(), weights.end(), 0.0);
	for (int i = 0; i < num_particles; ++i)
	{
		if (weight_sum != 0)
		{
			particles[i].weight /= weight_sum;
		}
		else
		{
			particles[i].weight = 0;
		}
	}
}

void ParticleFilter::resample() {
	// Description: This function randomly resamples the particles according to their weights.
	  // Create random number generator
	std::default_random_engine gen;

	// Create vector of weights
	vector<double> weights;
	for (int n = 0; n < num_particles; ++n)
	{
		weights.push_back(particles[n].weight);
	}

	// Creates discrete distribution according to weight vector
	std::discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> new_particles;

	// Choose new set of particles
	for (int n = 0; n < num_particles; ++n)
	{
		new_particles.push_back(particles[d(gen)]);
	}

	// Update particles
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
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
	vector<double> v;

	if (coord == "X") {
		v = best.sense_x;
	}
	else {
		v = best.sense_y;
	}

	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
