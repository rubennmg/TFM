# 18/11/2025

## Notes

-   [Basic Image Processing](https://www.cs.rice.edu/~vo9/recognition/notebooks/image_processing_lab.html)
-   PIL vs Rawpy

## TODOs

-   [x] Mejoras de escritura de código
-   [x] Utilizar método to `np_int8_array`
-   [x] Optimizar `image_loader`
    -   Nueva estructura para cargar diferentes formatos de imagen. Integración pendiente
-   [ ] `clamp_()` para disminuir cantidad de memoria reservada y copias de memoria
-   [ ] Optimizar tratamiento del tensor
-   [ ] Sesión de debug
-   [ ] Mirar HSV

# 19/11/2025

## Notes

## TODOs

-   [x] Carga de imágenes ARW
-   [x] Carga de imágenes DNG
-   [x] Carga de imágenes JPG
-   [x] Integración del nuevo `image_loader`

# 22/11/2025

## Notes

## TODOs

-   [x] Integración completa en `controller`
    -   [x] Revisión y debug
    -   [x] Mejora de tratamiento del tensor
    -   [x] Mejora en operaciones
        -   [x] `show_tensor` (`image_viewer`)
-   [x] Image model tiene un array de numpy int8 que se genera una vez se carga la imagen para visualización
-   [ ] Revisar signal rate en `right_panel`
-   [ ] Dilema de si actualizar o no el tensor original al debayerizar, esto genera un problema al aplicar mejora de contraste y desbayerizar
-   [ ] Mostrar nombre y formato de la imagen
-   [ ] Mostrar RGB, RAW, quizá `shape`...
-   [ ] Desacoplamiento de filtros
-   [ ] Tests
-   [ ] Modelo para perfil de configuración
