#include <stdio.h>
#include <unistd.h>    // for sleep()
#include <likwid.h>

int main() {
    // Initialize LIKWID Marker API
    likwid_markerInit();

    double a = 1.0;
    double b = 2.0;
    double c = 3.0;

    LIKWID_MARKER_START("Region1_DP");
    for (int i = 0; i < 100000000; i++) {
        // This is a Fused Multiply-Add (FMA) operation: (a * b) + c
        // A single FMA instruction counts as 2 FLOPs (1 mult, 1 add).
        a = (a * b) + c;
    }
    LIKWID_MARKER_STOP("Region1_DP");

    // Mark a region called "Region1"
    LIKWID_MARKER_START("Region1");
    for (int i = 0; i < 100000000; i++);  // simulate work
    LIKWID_MARKER_STOP("Region1");

    // Mark a second region called "Region2"
    LIKWID_MARKER_START("Region2");
    for (int i = 0; i < 50000000; i++);   // simulate shorter work
    LIKWID_MARKER_STOP("Region2");

    // Close LIKWID
    likwid_markerClose();

    return 0;
}
