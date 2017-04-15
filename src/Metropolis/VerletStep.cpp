#include "VerletStep.h"
#include "ProximityMatrixStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"

#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

//################################
// VerletStep definitions
//################################
Real VerletStep::calcMolecularEnergyContribution(int currMol, int startMol) {
    this->Contribution.CurrentMolecule( currMol );
    Real energy = 0.0;
    Real init = 0.0;
    int begIdx;
    int endIdx;

    if(GPUCopy::onGpu()) {

    } else {    // on CPU
        // Verlet list size check to handle initial system energy calculation
        if( this->h_verletList.size() != this->NUM_MOLS ) {
            begIdx = currMol * this->NUM_MOLS;
            endIdx = begIdx + this->NUM_MOLS;

            energy = thrust::transform_reduce( &this->h_verletList[begIdx],
                                               &this->h_verletList[endIdx],
                                               this->Contribution, init, this->sum);
        } else {
            // Asset verlet list size == NUM_MOLS
            energy = thrust::transform_reduce( this->h_verletList.begin(),
                                               this->h_verletList.end(),
                                               this->Contribution, init, this->sum);
            energy /= 2;    // To avoid double counting
        } // else CPU calcSystemEnergy
    } // else CPU
    return energy;
} // calcMolecularEnergyContribution

Real VerletStep::calcSystemEnergy( Real &subLJ, Real &subCharge, int numMolecules ) {

    // Set verlet list for initial system energy calculation
    // Sets each index in the list to its value
    // This is to use the index as molecule ID
    if( GPUCopy::onGpu() ) {
        this->d_verletList.resize( this->NUM_MOLS );
        thrust::sequence( thrust::device, this->d_verletList.begin(), this->d_verletList.end() );
    } else {
        this->h_verletList.resize( this->NUM_MOLS );
        thrust::sequence( thrust::host, this->h_verletList.begin(), this->h_verletList.end() );
    }

    Real result = SimulationStep::calcSystemEnergy( subLJ, subCharge, numMolecules );

    // Resize vector for verlet list use
    if( GPUCopy::onGpu() )
       this->d_verletList.resize( this->VERLET_SIZE );
    else
       this->h_verletList.resize( this->VERLET_SIZE );

    this->CreateNewVerletList();
    return result;
} // calcSystemEnergy

void VerletStep::checkOutsideSkinLayer(int molIdx) {
    //VerletCalcs::UpdateVerletList<int> update;
/*
    if( GPUCopy::onGpu() )
        if( update(molIdx, thrust::raw_pointer_cast(&d_verletAtomCoords[0]), GPUCopy::simBoxGPU()) )
            VerletStep::CreateVerletList();
    else    // on CPU
        if( update(molIdx, &h_verletAtomCoords[0], GPUCopy::simBoxCPU()) )
            VerletStep::CreateVerletList();
*/
} // checkOutsideSkinLayer

void VerletStep::changeMolecule( int molIdx, SimBox *box ) {
    SimulationStep::changeMolecule( molIdx, box );
    VerletStep::checkOutsideSkinLayer( molIdx );
} // changeMolecule

void VerletStep::rollback( int molIdx, SimBox *box ) {
    SimulationStep::rollback( molIdx, box );
    VerletStep::checkOutsideSkinLayer( molIdx );
} // rollback

VerletStep::~VerletStep() {
    this->h_verletList.clear();
    this->h_verletList.shrink_to_fit();
    this->h_verletAtomCoords.clear();
    this->h_verletAtomCoords.shrink_to_fit();

    this->d_verletList.clear();
    this->d_verletList.shrink_to_fit();
    this->d_verletAtomCoords.clear();
    this->d_verletAtomCoords.shrink_to_fit();
} // Destructor





//################################
// VerletCalcs definitions
//################################

template <typename T>
Real VerletCalcs::EnergyContribution<T>::operator()( const T neighbor ) const {
    Real total = 0.0;

    // Exit if neighbor is currMol or is not a verlet neighbor
    if( neighbor == -1 || neighbor == currMol )
        return total;

    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
    const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

    const int p2Start = molData[MOL_PIDX_START * numMolecules + neighbor];
    const int p2End = molData[MOL_PIDX_COUNT * numMolecules + neighbor] + p2Start;

    if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                atomCoords, bSize, pIdxes, cutoff, numAtoms))
        total += calcMoleculeInteractionEnergy(currMol, neighbor, sb);

    return total;
} // MoleculeContribution

template <typename T>
T* VerletCalcs::NewVerletList<T>::operator()() {
    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff * sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;
    int p1Start, p1End;
    int p2Start, p2End;

    int* verletList = new int[sb->numMolecules * sb->numMolecules];

    int numNeighbors;
    for(int i = 0; i < numMolecules; i++){

        numNeighbors = 0;
        p1Start = molData[MOL_PIDX_START * numMolecules + i];
        p1End = molData[MOL_PIDX_COUNT * numMolecules + i] + p1Start;

        for(int j = 0; j < numMolecules; j++){
            verletList[i * numMolecules + j] = -1;

            if (i != j) {
                p2Start = molData[MOL_PIDX_START * numMolecules + j];
                p2End = molData[MOL_PIDX_COUNT * numMolecules + j] + p2Start;

                if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                       atomCoords, bSize, pIdxes, cutoff, numAtoms)) {

                    verletList[i * numMolecules + numNeighbors ] = j;
                    numNeighbors++;
                } // if in range
            }
        } // for molecule j
    } // for molecule i
    return verletList;
} // newVerletList()







__host__ __device__
Real VerletCalcs::calcMoleculeInteractionEnergy(int m1, int m2, SimBox* sb) {
    Real energySum = 0;

    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    Real *aData = sb->atomData;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int m1Start = molData[MOL_START * numMolecules + m1];
    const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

    const int m2Start = molData[MOL_START * numMolecules + m2];
    const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

    for (int i = m1Start; i < m1End; i++) {
        for (int j = m2Start; j < m2End; j++) {
            if (aData[ATOM_SIGMA * numAtoms + i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
                && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

                const Real r2 = SimCalcs::calcAtomDistSquared(i, j, atomCoords, bSize, numAtoms);
                if (r2 == 0.0) {
                    energySum += 0.0;
                } else {
                    energySum += SimCalcs::calcLJEnergy(i, j, r2, aData, numAtoms);
                    energySum += SimCalcs::calcChargeEnergy(i, j, sqrt(r2), aData, numAtoms);
                }
            }
        }
    }
  return energySum;
}






/*

void VerletStep::CreateVerletList() {
    if( GPUCopy::onGpu() ) {
        VerletStep::freeMemory();
        this->d_verletList.resize( this->VERLET_SIZE );
        thrust::fill( this->d_verletList.begin(), this->d_verletList.end(), -1 );    // -1 as invalid molID
        this->d_verletAtomCoords.resize( this->VACOORDS_SIZE );

        // Copy starting verlet atom coordinates
        thrust::copy( this->d_verletAtomCoords.begin(),
                      this->d_verletAtomCoords.end(),
                      GPUCopy::simBoxGPU()->atomCoordinates );

        // Create and copy new verlet list
        VerletCalcs::NewVerletList<int> newNeighbors( GPUCopy::simBoxGPU() );
        thrust::copy( this->d_verletList.begin(), this->d_verletList.end(), 
                newNeighbors() );
    } else {    // on CPU
        VerletStep::freeMemory();
        this->h_verletList.resize( this->VERLET_SIZE );
        thrust::fill( this->h_verletList.begin(), this->h_verletList.end(), -1 );    // -1 as invalid molID
        this->h_verletAtomCoords.resize( this->VACOORDS_SIZE );

        // Copy starting verlet atom coordinates
        thrust::copy( this->h_verletAtomCoords.begin(),
                      this->h_verletAtomCoords.end(),
                      GPUCopy::simBoxCPU()->atomCoordinates );

        // Create and copy new verlet list
        VerletCalcs::NewVerletList<int> newNeighbors( GPUCopy::simBoxCPU() );
        thrust::copy( this->h_verletList.begin(), this->h_verletList.end(), 
                    newNeighbors() );
    } // else
} // CreateVerletList



// ----- VerletCalcs Definitions -----

template <typename T>
bool VerletCalcs::UpdateVerletList<T>::operator()( const T i, const Real* vaCoords, SimBox* sb ) const {
    const Real cutoff = sb->cutoff * sb->cutoff;
    Real* atomCoords = sb->atomCoordinates;
    Real* bSize = sb->size;
    int numAtoms = sb->numAtoms;

    Real dx = SimCalcs::makePeriodic(atomCoords[X_COORD * numAtoms + i] -  vaCoords[X_COORD * numAtoms + i], X_COORD, bSize);
    Real dy = SimCalcs::makePeriodic(atomCoords[Y_COORD * numAtoms + i] -  vaCoords[Y_COORD * numAtoms + i], Y_COORD, bSize);
    Real dz = SimCalcs::makePeriodic(atomCoords[Z_COORD * numAtoms + i] -  vaCoords[Z_COORD * numAtoms + i], Z_COORD, bSize);

    Real dist = pow(dx, 2) + pow(dy, 2) + pow(dz, 2);
    return dist > cutoff;
} // UpdateVerletList



*/

