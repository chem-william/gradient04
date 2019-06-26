Calculates the 4th-order accurate gradient of a 3-dimensional scalar function.
Uses central differences on the interior and first differences on boundaries to give the same shape as the input function.

This is to be used in calculating the current density through a molecule and has thus not been made general. Should be somewhat straightforward.

Originally it was implemented in Python as it is essentially an extension of how numpy.gradient works, but to increase performance and get more experience writing Rust, I decided to port it. It is probably horribly coded, but it initially gave me a 2X speed-up.
