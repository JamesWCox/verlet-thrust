#ifndef GPUCOPY_H
#define GPUCOPY_H

#include "SimBox.h"

namespace GPUCopy {

  bool onGpu();
  void setParallel(bool launchOnGpu);

  void copyIn(SimBox* sb);
  void copyOut(SimBox* sb);

  SimBox* simBoxGPU();
  SimBox* simBoxCPU();
}


#endif
