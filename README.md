
# neusci-daq-visualizer

Application designed to visualize recordings from data acquisition systems for students within the University of Washington's undergraduate neuroscience program. Designed for and tested with NI USB-6003 from National Instruments using PyDAQmx with the NIDAQmx driver. 

## Why/what?

<img width="1899" height="1092" alt="image" src="https://github.com/user-attachments/assets/c312c4dc-15bb-4d42-8993-7d0c08bb8a52" />

\

The UW undergraduate neuroscience program historically uses an A-M Systems 3000 AC/DC Differential Amplifier, a Humbug 60 Hz Noise Eliminator, and an AD Instruments PowerLab 4/26 Digitizer in concordance with LabChart 7 to record and analyze signals recorded from different model organisms. By doing so, students can learn about different information processing modalities in these model organisms to build neurobiology fundamentals. However, this system is dated and difficult to use on different operating systems. neusci-daq-visualizer is a flexible Python-based visualizer to transform signals into a computer-interpretable form and visualize it to make data analysis easy and efficient.

## Install (Windows: NI-DAQmx driver required)
1. Install the **NI-DAQmx** driver from ni.com. Check that NI MAX sees the device and note its name (`Dev1`, …).
2. Set up the environment:
```
   python -m venv .venv && .venv\Scripts\activate
   pip install -e ".[nidaqmx,dev]"
   python -m neusci_daq --list-devices
```
3. Edit `config.toml` (device name, channels, rate, scaling), then run:
```
   neusci-daq -c config.toml
   neusci-daq -c config.toml --backend simulated   # no hardware needed
```

## Usage

[wip]

## Methods

[wip]

## Who

Made for and by undergraduates within UW's neuroscience major!
