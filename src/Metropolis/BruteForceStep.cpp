#include "BruteForceStep.h"
#include "SimBoxConstants.h"
#include "SimulationStep.h"
#include "GPUCopy.h"


Real BruteForceStep::calcMolecularEnergyContribution(int currMol,
                                                     int startMol) {


  if (GPUCopy::onGpu()) {
    Real* devPtr;
    cudaMemcpy(&devPtr, &(GPUCopy::simBoxGPU()->atomCoordinates), sizeof(Real), cudaMemcpyDeviceToHost);

    BruteForceCalcs::calcMolecularEnergyContributionGPU<<<1,1>>>(currMol,
                                                                 startMol,
                                                                 GPUCopy::simBoxGPU());

    cudaMemcpy(&devPtr, &(GPUCopy::simBoxGPU()->atomCoordinates), sizeof(Real), cudaMemcpyDeviceToHost);

    Real energy;
    cudaDeviceSynchronize();
    cudaMemcpy(&energy, &(GPUCopy::simBoxGPU()->energy), sizeof(Real), cudaMemcpyDeviceToHost);
    return energy;
  } else {
    return BruteForceCalcs::calcMolecularEnergyContribution(currMol, startMol);
  }
}


// ----- BruteForceCalcs Definitions -----

__global__ 
void BruteForceCalcs::calcMolecularEnergyContributionGPU(int currMol,
                                                         int startMol,
                                                         SimBox* sb) {


  Real total = 0;

  int* molData = sb->moleculeData;
  Real* atomCoords = sb->atomCoordinates;
  Real* bSize = sb->size;
  int* pIdxes = sb->primaryIndexes;
  Real cutoff = sb->cutoff;
  const long numMolecules = sb->numMolecules;
  int numAtoms = sb->numAtoms;

  const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
  const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

  for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
    if (otherMol != currMol) {
      int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
      int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
      if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                     atomCoords, bSize, pIdxes, cutoff, numAtoms)) {
        total += calcMoleculeInteractionEnergy(currMol, otherMol, sb);
      }
    }
  }
  sb->energy = total;
}

Real BruteForceCalcs::calcMolecularEnergyContribution(int currMol,
                                                      int startMol) {
  Real total = 0;

  SimBox* sb = GPUCopy::simBoxCPU();
  int* molData = sb->moleculeData;
  Real* atomCoords = sb->atomCoordinates;
  Real* bSize = sb->size;
  int* pIdxes = sb->primaryIndexes;
  Real cutoff = sb->cutoff;
  const long numMolecules = sb->numMolecules;
  int numAtoms = sb->numAtoms;

  const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
  const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

  for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
    if (otherMol != currMol) {
      int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
      int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
      if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                     atomCoords, bSize, pIdxes, cutoff, numAtoms)) {
        total += calcMoleculeInteractionEnergy(currMol, otherMol, sb);
      }
    }
  }

  return total;
}

Real BruteForceCalcs::calcMoleculeInteractionEnergy (int m1, int m2, SimBox* sb) {
  Real energySum = 0;


  int* molData = sb->moleculeData;
  Real* aData = sb->atomData;
  Real* aCoords = sb->atomCoordinates;
  Real* bSize = sb->size;
  const long numMolecules = sb->numMolecules;
  const int numAtoms = sb->numAtoms;
  const int m1Start = molData[MOL_START * numMolecules + m1];
  const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

  const int m2Start = molData[MOL_START * numMolecules + m2];
  const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

  for (int i = m1Start; i < m1End; i++) {
    for (int j = m2Start; j < m2End; j++) {
      if (aData[ATOM_SIGMA * numAtoms +  i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
          && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

        const Real r2 = SimCalcs::calcAtomDistSquared(i, j, aCoords, bSize, numAtoms);
        if (r2 == 0.0) {
          energySum += 0.0;
        } else {
          energySum += SimCalcs::calcLJEnergy(i, j, r2, aData, numAtoms);
          energySum += SimCalcs::calcChargeEnergy(i, j, sqrt(r2), aData, numAtoms);
        }
      }
    }
  }

  return (energySum);
}
