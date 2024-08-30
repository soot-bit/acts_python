#!/usr/bin/env python3
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import time
import os

import acts
import acts.examples
from acts.examples.simulation import addParticleGun, addGeant4, EtaConfig
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.dd4hep import (
    DD4hepDetector,
    DD4hepDetectorOptions,
    DD4hepGeometryService,
)

u = acts.UnitConstants

def runGeant4( start_event, end_event, outputDir):
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    detector, trackingGeometry, _ = getOpenDataDetector()


    s = acts.examples.Sequencer(
        events=end_event - start_event, skip=start_event, numThreads=1 
    )

    s.config.logLevel = acts.logging.INFO
    rnd = acts.examples.RandomNumbers(seed=42)

    addParticleGun(
        s,
        EtaConfig(-2.0, 2.0),
        rnd=rnd,
    )
    o_dir = outputDir / f"process_{start_event}_{end_event}"
    o_dir.mkdir(parents=True, exist_ok=True)
    addGeant4(
        s,
        detector,
        trackingGeometry,
        field,
        outputDirCsv=o_dir,
        outputDirRoot=None,
        rnd=rnd,
    )
    s.run()

def chunkify(n_events, j):
    """Generate event chunks for j cores."""
    chunk_size = n_events // (j - 1)
    begins = list(range(0, n_events, chunk_size))
    ends = [min(b + chunk_size, n_events) for b in begins]
    return tuple(zip(begins, ends))


if __name__ == "__main__":

    s = time.time()
    j = max(cpu_count() - 2, 2)  # Reserve some CPUs
    chunks = chunkify(n_events=100, j=j)
    # breakpoint()
    worker = partial(runGeant4, outputDir = Path.cwd()) #(os.getenv("MOUNT_DIR")))

    with Pool(j) as p:
        p.starmap(worker, chunks)
        p.close()  
        p.join() 


    end_time = time.time()
    print(f"Total execution time: {end_time - s:.2f} s")
