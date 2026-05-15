# Scratch notes workspace for development

## 05/15/2026
One area where the assumption that "if sysfs, then nvidia-smi" breaks down is if the detected chip is 1000 series or earlier. A test case of this can be observed when running this program on tri-workstation: it has an NVIDIA GPU, specifically a 1080ti, and the sysfs check does, in fact, detect this properly. The problem is that nvidia has stopped supporting the 1000 series GPU with their latest driver updates starting with the 590 series drivers, meaning that nvidia-smi on this system is actually ahead of the driver version associated with the GPU. This breakdown is problematic, since the only way to remedy this (presumably) is to install an older version of nvidia-smi... which is just not a feasible approach.

Last week, Todd suggested we forego the sysfs check and just look directly for an nvidia-smi executable. Once again, this would not work in this case: this approach would detect that nvidia-smi is properly installed on tri-workstation, but the executable would fail with a version mismatch error.

The precise error is as follows:

```shell
(venv) aiden.mccormack@tri-workstation:~/archspec$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.71
```

### Potential workaround
It might be worth investigating the `lshw` command. If this is a general linux command (i.e. not specific to Ubuntu), then this might be a viable approach. This command effortlessly returned the correct GPU version for this system. See the example below:

```shell
(venv) aiden.mccormack@tri-workstation:~/archspec$ lshw -c video
WARNING: you should run this program as super-user.
  *-display
       description: VGA compatible controller
       product: GP102 [GeForce GTX 1080 Ti]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:01:00.0
       logical name: /dev/fb0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=nvidia latency=0 resolution=1920,1200
       resources: irq:85 memory:fa000000-faffffff memory:c0000000-cfffffff memory:d0000000-d1ffffff ioport:e000(size=128) memory:fb000000-fb07ffff
WARNING: output may be incomplete or inaccurate, you should run this program as super-user
```

### tl;dr
Using nvidia-smi is not viable to get chip information if the gpu is too old to receive the latest driver updates. Another approach must be used. Might be worth looking into `lspci`.
