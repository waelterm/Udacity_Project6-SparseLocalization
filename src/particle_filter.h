/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <string>
#include <vector>
#include "helper_functions.h"
struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};


class ParticleFilter {
public:
	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m],
	 *   standard deviation of y [m], standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
	 *   standard deviation of y [m], standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity,
		double yaw_rate);
	/*
	* multiVariateGaussian Calculates the Multi-Variate Gaussian
	* of an observation and its associated landmark
	* @param ox Standard deviation of x position of observation
	* @param oy Standard deviation of y position of observation
	* @param mu_x X position of observation
	* @param mu_y Y position of observation
	* @param x x position of associated landmark
	* @param y y position of associated landmark
	* Returns: Multi-variate Gaussian
	*/
	double multiVariateGaussian(double ox, double oy, double mu_x, double mu_y, double x, double y);

	/*
	* dataAssociation This function finds the closest landmark to an observation and returns its index
	* @param obs_x x coordinate of observation in map coordinates
	* @param obs_y y coordinate of observation in map coordinates
	* @param map_landmarks map including list of landmarks
	* @param range range of sensors in meters
	* Returns: Index of closest landmark to observation
	*/
	int dataAssociation(double obs_x, double obs_y, const Map& map_landmarks, double range);

	/*
	* vehicleToMap This function uses the position and heading of the particle to convert an observation from vehicle coordinates to map coordinates.
	* @param particle_x: x position associated with particle
	* @param particle_y: x position associated with particle
	* @param particle_theta: heading angle associated with particle
	* @param obs_x: x coordinate of observation in vehicle coordinates
	* @param obs_y: y coordinate of observation in vehicle coordinates
	* Returns: Vector of observation position in map coordinate system
	*/
	std::vector<double> vehicleToMap(double particle_x, double particle_y, double particle_theta, double obs_x, double obs_y);


	/**
	 * updateWeights Updates the weights for each particle based on the likelihood
	 *   of the observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2
	 *   [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs>& observations,
		const Map& map_landmarks);

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/**
	 * Set a particles list of associations, along with the associations'
	 *   calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations
	 *   are correct and assocations correctly connected
	 */
	void SetAssociations(Particle& particle, const std::vector<int>& associations,
		const std::vector<double>& sense_x,
		const std::vector<double>& sense_y);

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}

	/**
	 * Used for obtaining debugging information related to particles.
	 */
	std::string getAssociations(Particle best);
	std::string getSenseCoord(Particle best, std::string coord);

	// Set of current particles
	std::vector<Particle> particles;

private:
	// Number of particles to draw
	int num_particles;

	// Flag, if filter is initialized
	bool is_initialized;

	// Vector of weights of all particles
	std::vector<double> weights;
};

#endif  // PARTICLE_FILTER_H_
