## Requires:
 - `cloud`: organized strongest point cloud of one scan

## Ensure: 
Points of every Intesity peak `oeak result`

## Algorithm:
Filter the point cloud to horizontal plane of the sensor(z axis) save to `horizontal scan`;
**for** each `point` in `horizontal scan` **do**:
    calculate `gap` between this point and last;
    **if** `gap` > `threshhold` **then**:
        **if** the point is in a sequence increasing to the max intensity and decreasing sequence to next gap **then**:
            add the `point` to the sequence;
        **else**:
            **if** the sequence is valid **then**:
                store the peak to `potential peaks`;
**for** each `peak` in `potential peaks` **do**:
    choose points from same azimuth degree (same column) in the upper to lower two rings
    **if** verify the points is also an increasing then decreasing sequence **then**:
        store these points to `peak result` 