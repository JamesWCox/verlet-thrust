/**
 * VerletStep.h
 *
 * A subclass of SimulationStep that uses a "verlet list" for energy
 * calculations
 *
 */

#ifndef METROPOLIS_VERLETSTEP_H
#define METROPOLIS_VERLETSTEP_H

#include "SimulationStep.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class VerletStep: public SimulationStep {
    public:
        explicit VerletStep(SimBox* box): SimulationStep(box),
                                    h_verletList(0),
                                    h_verletAtomCoords(0) {}
//                                    d_verletList(0),
//                                    d_verletAtomCoords(NULL) {}

        virtual ~VerletStep();
        virtual Real calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules);

       /**
        * Determines the energy contribution of a particular molecule.
        * @param currMol The index of the molecule to calculate the contribution
        * @param startMol The index of the molecule to begin searching from to 
        *                      determine interaction energies
        * @param verletList The host_vector<int> containing the indexes of molecules
        *                      in range for each molecule 
        * @return The total energy of the box (discounts initial lj / charge energy)
        */
        virtual Real calcMolecularEnergyContribution(int currMol, int startMol);
        virtual void changeMolecule(int molIdx, SimBox *box);
        virtual void rollback(int molIdx, SimBox *box);

    private:
        thrust::host_vector<int> h_verletList;
        thrust::host_vector<Real> h_verletAtomCoords;
//        thrust::device_vector<int> d_verletList;
//        Real* d_verletAtomCoords;
        void createVerlet();
};

/**
 * VerletCalcs namespace
 *
 * Contains logic for calculations used by the VerletStep class.
 * Although logically related to the VerletStep class, these need
 * to be seperated to accurately run on the GPU.
 */
namespace VerletCalcs {
        
    /**
     * Determines the energy contribution of a particular molecule.
     * @param currMol The index of the molecule to calculate the contribution
     * @param startMol The index of the molecule to begin searching from to 
     *                      determine interaction energies
     * @param verletList The host_vector<int> containing the indexes of molecules
     *                      in range for each molecule 
     * @return The total energy of the box (discounts initial lj / charge energy)
     */
    template <typename T>
    struct calcMolecularEnergyContribution {
        __host__ //__device__
        Real operator()(int currMol, int startMol, thrust::host_vector<int> verletList) const;
    };
    
     /**
      * Determines whether or not two molecule's primaryIndexes are
      * within the cutoff range of one another and calculates the 
      * energy between them (if within range)
      *
      * @param m1 Molecule 1 
      * @param m2 Molecule 2
      * @param sb The SimBox from which data is to be used 
      * @return The total energy between two molecules 
      */
    __host__ __device__
    Real calcMoleculeInteractionEnergy (int m1, int m2, SimBox* sb);
    
    /**
     * Checks if the verlet list needs to be updated to account for 
     * changes to molecule positions
     *
     * @return True/False if an update needs to take place
     */
    bool updateVerlet(thrust::host_vector<Real> &vaCoords, int i);
    
    /**
     * Frees memory used for a CPU run
     */
    void freeMemory(thrust::host_vector<int> &h_verletList, thrust::host_vector<Real> &vaCoords);

    /**
     * Creates a new verlet list for CPU run
     *
     * @return A host_vector<int> representing a verlet list
     */
    thrust::host_vector<int> newVerletList();
}

#endif
