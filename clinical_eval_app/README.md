# Clinical Eval App

Aplicacion web estatica para validacion clinica de justificaciones de IA (Match / No Match), sin backend.

## Objetivo

Permitir que evaluadores medicos revisen casos en local, registren veredictos y exporten resultados en CSV para analisis posterior.

## Estructura esperada

- index.html
- style.css
- app.js
- cases.json
- images/

Tambien puedes trabajar con datasets alternativos (por ejemplo exportados por escenario) cargandolos manualmente desde el Gestor de Data. La app los conserva en una biblioteca local y te deja cambiar el case activo desde el selector inferior.

Importante: `image_path` debe ser relativo a la carpeta donde vive cada `cases.json`. En los exportes nuevos el contrato es `dataset_root` + `images/<archivo>` y la app reconstruye la ruta completa a partir de esa raiz local.

## Esquema de entrada (cases.json)

Cada caso debe seguir este formato:

```json
[
  {
    "id_imagen": "polyp_042",
    "image_path": "images/polyp_042_bbox.jpg",
    "ground_truth_class": "AD",
    "ai_predicted_class": "AD",
    "id_modelo_oculto": "qwen3.5-9b",
    "clinical_justification": "La lesion presenta un patron vascular irregular..."
  }
]
```

Notas:

- id_modelo_oculto no se muestra en la UI (doble ciego), pero se usa en el CSV exportado.
- image_path debe apuntar a la imagen desde la raiz del bundle donde esta el `cases.json`.
- Si el export incluye `dataset_root`, la app usa esa raiz para resolver la imagen de forma local sin exponer rutas absolutas del sistema.
- En los bundles exportados, lo normal es que `image_path` tenga la forma `images/<archivo>`.

## Flujo principal

1. La app carga la biblioteca de datasets guardada en `localStorage` y restaura el case activo si existe.
2. Muestra un caso por pantalla con imagen, diagnostico real y justificacion.
3. El evaluador vota Match o No Match y puede anadir comentario opcional.
4. Se puede navegar con flechas, abrir la galeria o cambiar el case activo desde el selector inferior.
5. El progreso se guarda por dataset en `localStorage`.
6. El CSV se puede exportar durante la evaluacion o al finalizar.

## Pantalla inicial y carga local

Si no hay dataset activo, la app muestra la pantalla inicial de carga.

- Desde ahi se importa cases.json manualmente.
- Al importar, la app añade el nuevo dataset a la biblioteca local y lo deja activo sin borrar los anteriores.
- En esa pantalla no se muestra accion de borrado porque aun no hay un dataset activo.

## Selector de Case

En la barra inferior hay un desplegable de "Caso activo" con todos los datasets guardados.

- Al cambiar de opcion, la app restaura ese dataset y su progreso propio.
- El selector solo cambia el dataset activo, no borra ningun dato.
- Si un dataset tiene resultados y comentarios guardados, se conservan al volver a seleccionarlo.

## Gestor de Data

El Gestor de Data (boton "Gestor de Data" en la barra inferior) permite:

- Importar un nuevo cases.json.
- Borrar solo el case/dataset activo.

Importante: "Borrar case actual" elimina solo el dataset seleccionado y deja intactos los demas datasets guardados.

## Galeria interactiva

La galeria incluye filtros interactivos por estado:

- Match, No Match y Pendiente: filtro exclusivo (solo uno activo a la vez).
- Con comentario: filtro adicional combinable con cualquiera de los anteriores.

Las etiquetas de filtro tienen tamano homogeneo para mejorar lectura y consistencia visual.

## Manejo de data faltante o rota

Si la app detecta que la data actual fue borrada o que faltan imagenes referenciadas:

- avisa del problema en el case activo,
- permite cambiar a otro dataset desde el selector,
- muestra un mensaje de error explicando el problema.

Esto evita quedarse en un estado intermedio sin informacion visible.

## Persistencia local

Se guardan en `localStorage`:

- biblioteca de datasets guardados,
- dataset activo,
- resultados acumulados por dataset,
- indice del caso actual por dataset,
- comentarios borrador por caso.

Si se cierra y reabre la pagina, la evaluacion se reanuda con el dataset activo y el progreso asociado cuando existe data valida en cache.

## Exportacion CSV

Cabecera exacta:

```text
id_imagen,id_modelo_oculto,veredicto,comentario_medico
```

Nombre de descarga:

- evaluacion_clinica_resultados.csv

## Uso rapido

1. Coloca `cases.json` e `images/` en la misma raiz de exportacion, o importa un bundle ya generado.
2. Genera o edita `cases.json` con `image_path` relativo a esa raiz.
3. Abre `index.html` en el navegador.
4. Importa datasets nuevos desde el Gestor de Data y usa el selector de caso activo para cambiar entre ellos.
5. Evalua casos y exporta CSV cuando quieras.