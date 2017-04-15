/**
 *  ParallelCalcs.h
 */
#ifndef PARALLELCALCS_H
#define PARALLELCALCS_H
#include "Metropolis/SimBox.h"

namespace ParallelCalcs {

    /**
     * Returns the long-range energy correction from the device.
     * Can be easily modified to handle CPU-side computation.
     */
    Real calcEnergy_LRC();

    /**
     * CUDA kernel for long-range correction calculations
     * @param sb A SimBox containing all the molecule data
     */
    __global__
    void LRC_Kernel(SimBox* sb);

    /**
     * Calculates the long-range correction energy value for molecules outside the cutoff.
     * @param sb A SimBox containing all the molecule data
     */
    __host__ __device__
    void calculateLRC(SimBox* sb);
}
#endif


