"""Native-only, pinned single-operation RecipeSpecs for extended thresholds."""
from dataclasses import replace
from types import MappingProxyType

from seedvision.benchmarking.recipe_catalogue import (
    _image_semantics, _load_source_config, _mask_semantics,
    _plantcv_input_stage, _plantcv_mapping, _plantcv_provenance,
)
from seedvision.benchmarking.recipes import (
    DependencyContract, LayoutPosition, ParameterContract, PortContract,
    RecipeProvenanceMode, RecipeSpec, RecipeStage,
)
from seedvision.cuda.plantcv_extended_thresholds import (
    CHANNELS, TEXTURE_BORDERS, TEXTURE_PROPERTIES,
)

NATIVE_EXTENDED_THRESHOLD_VERSION = "1.1.0"
_THRESHOLD_SOURCE = "plantcv/plantcv/threshold/threshold_methods.py"
_SOURCE_SYMBOLS = {
    "threshold.gaussian": "plantcv.plantcv.threshold.threshold_methods.gaussian",
    "threshold.triangle": "plantcv.plantcv.threshold.threshold_methods.triangle",
    "threshold.texture": "plantcv.plantcv.threshold.threshold_methods.texture",
    "threshold.dual_channels": "plantcv.plantcv.threshold.threshold_methods.dual_channels",
}
_TITLES = {
    "threshold.gaussian": "PlantCV Gaussian threshold",
    "threshold.triangle": "PlantCV triangle threshold",
    "threshold.texture": "PlantCV texture threshold",
    "threshold.dual_channels": "PlantCV dual-channel threshold",
}


def _parameter_contracts():
    # Construct these values from verified function code, rather than retaining
    # mutable dataclass-valued globals outside the loaded-code recorder.
    return {
        "threshold.gaussian": (
            ParameterContract("ksize", "integer", 3, True, editable=True, minimum=3, maximum=15,
                description="Odd Gaussian kernel from 3 to 15; uncertain byte-rounding results are unavailable."),
            ParameterContract("offset", "number", 0.0, True, editable=True, units="intensity",
                minimum=-255, maximum=255, description="Finite adaptive threshold offset in [-255, 255]."),
            ParameterContract("object_type", "string", "light", True, editable=True,
                choices=("light", "dark"), description="Foreground polarity."),
        ),
        "threshold.triangle": (
            ParameterContract("object_type", "string", "light", True, editable=True,
                choices=("light", "dark"), description="Foreground polarity."),
        ),
        "threshold.texture": (
            ParameterContract("ksize", "integer", 5, True, editable=True, minimum=3, maximum=9,
                description="Odd supported GLCM kernel: 3, 5, 7, or 9."),
            ParameterContract("offset", "integer", 1, True, editable=True, minimum=1, maximum=8,
                description="Positive GLCM offset, smaller than the selected kernel."),
            ParameterContract("texture_method", "string", "dissimilarity", True, editable=True,
                choices=TEXTURE_PROPERTIES, description="GLCM property used for the byte decision."),
            ParameterContract("borders", "string", "nearest", True, editable=True,
                choices=TEXTURE_BORDERS, description="Pinned SciPy neighborhood border mode."),
            ParameterContract("threshold", "integer", 32, True, editable=True, minimum=0, maximum=255,
                units="byte", description="Strict binary cutoff in [0, 255]."),
        ),
        "threshold.dual_channels": (
            ParameterContract("x_channel", "string", "R", True, editable=True,
                choices=tuple(CHANNELS), description="Case-sensitive first PlantCV channel name."),
            ParameterContract("y_channel", "string", "G", True, editable=True,
                choices=tuple(CHANNELS), description="Case-sensitive second PlantCV channel name."),
            ParameterContract("points", "json", [[0, 0], [255, 255]], True, editable=True,
                description="Ordered numeric x/y points; the pinned operation uses its first two points."),
            ParameterContract("above", "boolean", True, True, editable=True,
                description="Apply the strict above-line rule when true, below-line rule when false."),
        ),
    }


def build_native_extended_threshold_recipes():
    """Build four separate working recipes outside the frozen reference catalogue."""
    config = _load_source_config()
    original = _plantcv_provenance(config)
    provenance = replace(
        original,
        mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id="seedvision.cuda.plantcv_extended_thresholds",
        backend_version=NATIVE_EXTENDED_THRESHOLD_VERSION,
        evidence_note=(
            "Native CUDA single-operation adapter for a pinned PlantCV 4.11.3 source. "
            "The adapter has only the per-operation domains declared by its implementation; "
            "this identity does not claim broader reference parity or biological validity."
        ),
    )
    contracts, recipes = _parameter_contracts(), {}
    for operation in (
        "threshold.gaussian", "threshold.dual_channels", "threshold.texture", "threshold.triangle"
    ):
        recipe_id = "plantcv_" + operation.replace(".", "_")
        external = _plantcv_input_stage(recipe_id)
        is_rgb = operation == "threshold.dual_channels"
        input_name = "rgb" if is_rgb else "gray"
        input_type = "RGBImage" if is_rgb else "GrayImage"
        semantics = (
            {**_image_semantics("RGB"), "channel_boundary": "RGB uint8 HWC"}
            if is_rgb else _image_semantics("GRAY")
        )
        stage = RecipeStage(
            stage_id=recipe_id + "_threshold",
            title=_TITLES[operation],
            operation_id=operation,
            description=(
                "Native CUDA implementation of the pinned PlantCV threshold operation. "
                "The caller supplies decoded RGB and grayscale boundaries; only the "
                "selected operation runs and returns a uint8 binary mask. Unsupported or "
                "numerically uncertain domains fail explicitly. Starting parameter values "
                "are working selections, not validated acquisition settings."
            ),
            ports=(
                PortContract(input_name, "input", input_type,
                    "RGB image" if is_rgb else "Grayscale image", semantics=semantics),
                PortContract("mask", "output", "BinaryMask", "Binary foreground mask",
                    semantics=_mask_semantics()),
            ),
            parameters=contracts[operation],
            source_mappings=(_plantcv_mapping(
                config, _THRESHOLD_SOURCE, _SOURCE_SYMBOLS[operation],
                note="Pinned PlantCV 4.11.3 operation source; native domains are stated separately.",
            ),),
        )
        edge = DependencyContract(external.stage_id,
            "rgb_image" if is_rgb else "gray_image", stage.stage_id, input_name, input_type)
        recipes[recipe_id] = RecipeSpec(
            recipe_id=recipe_id,
            recipe_version=NATIVE_EXTENDED_THRESHOLD_VERSION,
            title=_TITLES[operation],
            description=(
                "Native single-operation threshold recipe. This additive working recipe "
                "does not alter the historical catalogue or claim a complete seed workflow."
            ),
            provenance=provenance,
            stages=(external, stage),
            dependencies=(edge,),
            layout=(LayoutPosition(external.stage_id, 0.0, 0.0),
                    LayoutPosition(stage.stage_id, 260.0, 0.0)),
        )
    return MappingProxyType(recipes)


__all__ = ["NATIVE_EXTENDED_THRESHOLD_VERSION", "build_native_extended_threshold_recipes"]
