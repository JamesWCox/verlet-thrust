/**
 * @file Application.cpp
 *
 *  Contains the main entry point for the program and the method to start
 *  running the simulation.
 *
 * @author Tavis Maclellan
 * @date Created 2/23/2014
 * @date Updated 1/7/2016
 */

#include "Application.h"
#include <iostream>
#include <fstream>


int metrosim::run(int argc, char** argv) {
	SimulationArgs args = SimulationArgs();

	if (!getCommands(argc, argv, &args)) {
		exit(EXIT_FAILURE);
	}

  if (args.simulationMode == SimulationMode::Parallel) {
    GPUCopy::setParallel(true);
  } else {
    args.simulationMode = SimulationMode::Serial;
    fprintf(stdout, "Beginning simulation using CPU...\n");
  }

	Simulation sim = Simulation(args);
	sim.run();

	fprintf(stdout, "Finishing simulation...\n\n");

// Shutdown the device if we were using the GPU
#ifdef _OPENACC
	if (args.simulationMode == SimulationMode::Parallel) {
		acc_shutdown(acc_device_nvidia);
	}
#endif

	exit(EXIT_SUCCESS);
}
