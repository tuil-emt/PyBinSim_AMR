This Folder must inherit a MAT file including a strutured list of all directivity dependend binaural room impulse responses. An example can be found here: https://cloud.tu-ilmenau.de/s/sdPgjXyxcyBWdDC

This MAT file must be specified by name in the config file in the root directory. It can be opened and modified using Matlab. It lists each part of the BRIR with direct source (DS), early reflections (ER), and late reverb (LR) for each direction around the head as well as a head phone equalization filter (only necessary when activated in the config file). The basic structure Looks like the following: 


root
\hp (7xF struct) … F..number of headphone filters
\\ struct(1)
\\\ type (char)
\\\ listenerOrientation (triple double) … i.e.: [0,0,0] azimuth, elevation, tilt
\\\ listenerPosition (triple double)
\\\ sourceOrientation (triple double)
\\\ sourcePosition (triple double)
\\\ custom (triple double)
\\\ filter (Nx2 single vector) … 2 channel filter response of length N
\binsim (7xB struct) … B..number of binaural entries
\\ struct(1)
\\\ type (char) … i.e.: 'DS', or 'ER'
\\\ listenerOrientation (triple double) … i.e.: [0,0,0] azimuth, elevation, tilt
\\\ listenerPosition (triple double)
\\\ sourceOrientation (triple double)
\\\ sourcePosition (triple double)
\\\ custom (triple double)
\\\ filter (Nx2 single vector) … 2 channel filter response of length N
.
.
.
\\ struct(B)
\\\ type (char) … i.e.: 'LR', or 'ER'
\\\ listenerOrientation (triple double) … i.e.: [-180,50,0] azimuth, elevation, tilt
\\\ listenerPosition (triple double)
\\\ sourceOrientation (triple double)
\\\ sourcePosition (triple double)
\\\ custom (triple double)
\\\ filter (Nx2 single vector) … 2 channel filter response of length N

