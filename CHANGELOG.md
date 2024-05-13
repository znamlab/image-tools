# Change log

## 13/05/2024

- Bugfix: affine_by_block could fit noise if too many blocks were used. Add a threshold
    in precentage of valid blocks to help avoid this case.

## 04/04/2024

- Add affine_by_block module: affine registration from running phase correlation in
    blocks of the image.
