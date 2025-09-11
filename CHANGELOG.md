# Change log

## 11/09/2025

- add `eps` when whitening to avoid divid by 0 warnings and NaNs.

## 08/08/2024

- Feature: add whiten argument in affine by block

## 13/05/2024

- Bugfix: affine_by_block could fit noise if too many blocks were used. Add a threshold
    in percentage of valid blocks to help avoid this case.

## 04/04/2024

- Add affine_by_block module: affine registration from running phase correlation in
    blocks of the image.
