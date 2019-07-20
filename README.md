Calculates the 4th-order accurate gradient of a 3-dimensional scalar function.
Uses central differences on the interior and first differences on boundaries to give the same shape as the input function.

To be used in calculating the current density through a molecule and has thus not been made general, but that should be somewhat straightforward.

Originally it was implemented in Python as it is essentially a port of numpy.gradient, but to increase performance and get more experience writing Rust, I decided to port it. It is probably horribly coded, but it gave me almost 3x speedup.
