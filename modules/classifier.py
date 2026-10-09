from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ModelLoadError(RuntimeError):
    pass


class ModelNotFoundError(ModelLoadError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    index: int
    probabilities: np.ndarray = field(repr=False)


@dataclass
class ModelMetadata:
    input_shape: tuple[int | None, ...] | None = None
    output_shape: tuple[int | None, ...] | None = None
    labels: dict[int, str] = field(default_factory=dict)
    model_info: dict[str, Any] = field(default_factory=dict)
    source: str = ""


class GestureClassifier:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        model_info_path: Path,
        model_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.model_info_path = Path(model_info_path)
        self.model_dir = Path(model_dir)
        self.logger = logger
        self.model = None
        self.metadata = ModelMetadata()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        resolved_model_path = self._resolve_model_path()
        load_source = ""

        try:
            self.model, load_source = self._load_keras_model(resolved_model_path)
            self._inject_runtime_globals(self.model)
        except Exception as exc:
            if not self._is_weights_only_file(resolved_model_path):
                raise ModelLoadError(f"Unable to load model: {resolved_model_path}") from exc

            output_classes = self._preferred_output_classes(resolved_model_path)
            self.model = self._build_reference_cnn(output_classes=output_classes)
            try:
                self.model.load_weights(resolved_model_path)
            except Exception as weight_exc:
                raise ModelLoadError(
                    "The .h5 file appears to contain weights only, but they do not match the reference CNN architecture"
                ) from weight_exc
            load_source = "reference_cnn + weights"

        self.metadata.input_shape = self._shape_tuple(getattr(self.model, "input_shape", None))
        self.metadata.output_shape = self._shape_tuple(getattr(self.model, "output_shape", None))
        self.metadata.model_info = self._load_model_info()
        self.metadata.labels = self._load_labels()
        self.metadata.source = load_source or "serialized_model"
        self._validate_label_count()
        self._validate_image_model()

        if self.logger:
            self.logger.info(
                "Loaded model %s with input_shape=%s output_shape=%s labels=%s source=%s",
                resolved_model_path,
                self.metadata.input_shape,
                self.metadata.output_shape,
                len(self.metadata.labels),
                self.metadata.source,
            )

    def predict(self, features: np.ndarray) -> Prediction:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        outputs = self.model.predict(features, verbose=0)
        if isinstance(outputs, list):
            outputs = outputs[0]
        probabilities = np.asarray(outputs, dtype=np.float32)
        if probabilities.ndim > 1:
            probabilities = probabilities[0]
        probabilities = probabilities.reshape(-1)
        if probabilities.size == 0:
            raise RuntimeError("Model returned an empty prediction")

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = self.metadata.labels.get(index, str(index))
        return Prediction(label=label, confidence=confidence, index=index, probabilities=probabilities)

    def _load_keras_model(self, resolved_model_path: Path):
        serialized_error: Exception | None = None
        if self._has_serialized_model_config(resolved_model_path):
            try:
                return self._load_serialized_h5_model(resolved_model_path), "serialized_h5_model"
            except Exception as exc:
                serialized_error = exc
                if self.logger:
                    self.logger.warning("Structured H5 load failed, trying generic Keras loader: %s", exc)

        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:
            raise ModelLoadError("TensorFlow/Keras is not installed or could not be imported") from exc

        try:
            from keras import config as keras_config  # type: ignore

            if hasattr(keras_config, "enable_unsafe_deserialization"):
                keras_config.enable_unsafe_deserialization()
        except Exception:
            pass

        try:
            return load_model(resolved_model_path, compile=False, safe_mode=False), "keras_load_model"
        except TypeError:
            return load_model(resolved_model_path, compile=False), "keras_load_model"
        except Exception as exc:
            if serialized_error is not None:
                raise serialized_error from exc
            raise

    def _load_serialized_h5_model(self, resolved_model_path: Path):
        try:
            import h5py
            import tensorflow as tf
            from keras import config as keras_config  # type: ignore
            from keras.src.models.functional import Functional
        except Exception as exc:
            raise ModelLoadError("Required Keras/HDF5 components are unavailable") from exc

        if hasattr(keras_config, "enable_unsafe_deserialization"):
            keras_config.enable_unsafe_deserialization()

        with h5py.File(resolved_model_path, "r") as file:
            raw_config = file.attrs.get("model_config")
            if raw_config is None:
                raise ModelLoadError("The model file does not contain a serialized model_config")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            model_config = json.loads(raw_config)

        self._patch_serialized_model_config(model_config)
        model_json = json.dumps(model_config)
        model = tf.keras.models.model_from_json(model_json, custom_objects={"Functional": Functional})
        model.load_weights(resolved_model_path)
        self._inject_runtime_globals(model)
        return model

    def _inject_runtime_globals(self, model: Any) -> None:
        try:
            import tensorflow as tf
        except Exception:
            return

        layers = getattr(model, "layers", []) or []
        for layer in layers:
            function = getattr(layer, "function", None)
            if callable(function):
                try:
                    function.__globals__.setdefault("tf", tf)
                except Exception:
                    pass
            sublayers = getattr(layer, "layers", None)
            if sublayers:
                self._inject_runtime_globals(layer)

    def _patch_serialized_model_config(self, model_config: dict[str, Any]) -> None:
        layers = model_config.get("config", {}).get("layers", [])
        for layer in layers:
            if layer.get("class_name") != "Lambda":
                continue
            config = layer.setdefault("config", {})
            if config.get("output_shape"):
                continue
            output_shape = self._infer_lambda_output_shape(layer)
            if output_shape is not None:
                config["output_shape"] = output_shape

    def _infer_lambda_output_shape(self, layer: dict[str, Any]) -> list[int] | None:
        name = str(layer.get("name", "")).strip().lower()
        inbound_shape = self._find_first_tensor_shape(layer.get("inbound_nodes"))
        if not inbound_shape:
            return None

        dims = [None if dim is None else int(dim) for dim in inbound_shape]
        if len(dims) < 4:
            return None

        height = dims[1]
        width = dims[2]
        channels = dims[3]
        if height is None or width is None:
            return None

        if name == "gray_to_rgb":
            return [int(height), int(width), 3]

        if name == "mobilenet_preprocess":
            if channels is None:
                return None
            return [int(height), int(width), int(channels)]

        return [int(dim) for dim in dims[1:] if dim is not None]

    def _find_first_tensor_shape(self, value: Any) -> tuple[int | None, ...] | None:
        if isinstance(value, dict):
            if value.get("class_name") == "__keras_tensor__":
                config = value.get("config", {})
                shape = config.get("shape")
                if isinstance(shape, (list, tuple)):
                    return tuple(None if dim is None else int(dim) for dim in shape)
                return None
            for item in value.values():
                found = self._find_first_tensor_shape(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._find_first_tensor_shape(item)
                if found is not None:
                    return found
        return None

    def _build_reference_cnn(self, output_classes: int):
        try:
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
        except Exception as exc:
            raise ModelLoadError("TensorFlow/Keras layers could not be imported") from exc

        model = Sequential(
            [
                Input(shape=(128, 128, 1)),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.4),
                Dense(96, activation="relu"),
                Dropout(0.4),
                Dense(64, activation="relu"),
                Dense(max(1, int(output_classes)), activation="softmax"),
            ]
        )
        return model

    def _preferred_output_classes(self, weights_path: Path) -> int:
        inferred = self._infer_output_classes_from_weights(weights_path)
        if inferred:
            return inferred

        labels = self._load_labels(strict=False)
        if labels:
            return len(labels)

        info = self._load_model_info()
        if isinstance(info.get("num_classes"), int) and info["num_classes"] > 0:
            return int(info["num_classes"])
        return 35

    def _infer_output_classes_from_weights(self, path: Path) -> int | None:
        try:
            import h5py
        except Exception:
            return None

        try:
            with h5py.File(path, "r") as file:
                layer_names = file.attrs.get("layer_names")
                if layer_names is None:
                    return None
                decoded_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in layer_names]
                for layer_name in reversed(decoded_names):
                    if layer_name not in file:
                        continue
                    layer_group = file[layer_name]
                    for dataset in self._iter_datasets(layer_group):
                        if len(dataset.shape) == 2:
                            return int(dataset.shape[1])
        except Exception:
            return None
        return None

    def _iter_datasets(self, group) -> Iterator[Any]:
        for value in group.values():
            if hasattr(value, "shape"):
                yield value
            elif hasattr(value, "values"):
                yield from self._iter_datasets(value)

    def _resolve_model_path(self) -> Path:
        if self.model_path.exists():
            return self.model_path
        candidates = sorted(self.model_dir.glob("*.h5")) + sorted(self.model_dir.glob("*.keras"))
        if candidates:
            if self.logger:
                self.logger.warning("Using discovered model file: %s", candidates[0])
            return candidates[0]
        raise ModelNotFoundError(
            f"No model file found. Add {self.model_path.name} or another .h5/.keras file to {self.model_dir}."
        )

    def _has_serialized_model_config(self, path: Path) -> bool:
        try:
            import h5py

            with h5py.File(path, "r") as file:
                return "model_config" in file.attrs
        except Exception:
            return False

    def _is_weights_only_file(self, path: Path) -> bool:
        try:
            import h5py

            with h5py.File(path, "r") as file:
                return "model_config" not in file.attrs and "layer_names" in file.attrs
        except Exception:
            return False

    def _load_model_info(self) -> dict[str, Any]:
        if not self.model_info_path.exists():
            return {}
        try:
            with self.model_info_path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            if self.logger:
                self.logger.warning("Failed to read model_info.json: %s", exc)
            return {}

    def _load_labels(self, strict: bool = True) -> dict[int, str]:
        output_count = self._output_count()
        fallback = {index: str(index) for index in range(output_count)} if output_count else {}

        if not self.labels_path.exists():
            if strict and self.logger:
                self.logger.warning("labels.json missing; falling back to numeric labels")
            return fallback

        try:
            raw = json.loads(self.labels_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            if strict and self.logger:
                self.logger.warning("Failed to read labels.json; falling back to numeric labels: %s", exc)
            return fallback

        labels = self._coerce_labels(raw)
        if labels:
            return labels

        if strict and self.logger:
            self.logger.warning("labels.json is not a label map; falling back to numeric labels")
        return fallback

    def _coerce_labels(self, raw: Any) -> dict[int, str]:
        if isinstance(raw, list):
            return {index: str(label) for index, label in enumerate(raw)}

        if isinstance(raw, dict):
            if {"class_name", "config", "backend", "keras_version"}.issubset(raw.keys()):
                return {}
            if "labels" in raw:
                nested = raw["labels"]
                if isinstance(nested, list):
                    return {index: str(label) for index, label in enumerate(nested)}
                if isinstance(nested, dict):
                    raw = nested
            labels: dict[int, str] = {}
            for key, value in raw.items():
                try:
                    labels[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            return labels

        return {}

    def _validate_label_count(self) -> None:
        output_count = self._output_count()
        if output_count and len(self.metadata.labels) != output_count and self.logger:
            self.logger.warning(
                "Label count (%s) does not match model output count (%s)",
                len(self.metadata.labels),
                output_count,
            )

    def _validate_image_model(self) -> None:
        shape = self.metadata.input_shape
        if not shape:
            return
        if len(shape) != 4 or shape[-1] != 1:
            if self.logger:
                self.logger.warning("Expected grayscale CNN input, got %s", shape)

    def _output_count(self) -> int:
        shape = self.metadata.output_shape
        if not shape:
            return 0
        last = shape[-1]
        return int(last) if last is not None else 0

    def _shape_tuple(self, shape: Any) -> tuple[int | None, ...] | None:
        if shape is None:
            return None
        if isinstance(shape, list) and shape and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        try:
            return tuple(None if dim is None else int(dim) for dim in shape)
        except TypeError:
            return None


