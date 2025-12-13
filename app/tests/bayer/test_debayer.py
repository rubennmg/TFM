import pytest
import torch
from torch import Tensor

from core.bayer.debayer import Debayer
from enums.layouts import Layout


class TestDebayer:
    """Unit tests for the Debayer operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the debayering operation for various algorithms

    See Also:
        core.bayer.debayer.Debayer: The Debayer operation class being tested
    """

    def test_parameter_validation(self) -> None:
        algorithm_name: str = "debayer3x3"
        layout: Layout = Layout.RGGB

        op = Debayer(algorithm_name=algorithm_name, layout=layout)

        assert isinstance(op.algorithm_name, str)
        assert op.algorithm_name == algorithm_name

        assert isinstance(op.layout, Layout)
        assert op.layout == layout

        invalid_algorithm_name: str = "invalid_debayer"

        with pytest.raises(ValueError):
            Debayer(algorithm_name=invalid_algorithm_name, layout=layout)

    @pytest.mark.parametrize(
        "algorithm_name,layout",
        [
            ("debayer2x2", Layout.RGGB),
            ("debayer3x3", Layout.RGGB),
            ("debayer5x5", Layout.RGGB),
            ("debayersplit", Layout.RGGB),
        ],
    )
    def test_apply_operation(
        self, base_tensor, assert_tensors, assert_channels, algorithm_name, layout
    ) -> None:
        entry = base_tensor(shape=[1, 1, 4, 4], dtype="float")
        op = Debayer(algorithm_name=algorithm_name, layout=layout)

        result = op(entry)

        assert_tensors(result, self.expected_debayer(algorithm_name=algorithm_name))
        assert_channels(result, expected_channels=3)

    def expected_debayer(self, algorithm_name: str) -> Tensor:
        if algorithm_name == "debayer2x2":
            return expected_debayer2x2()
        elif algorithm_name == "debayer3x3":
            return expected_debayer3x3()
        elif algorithm_name == "debayer5x5":
            return expected_debayer5x5()
        elif algorithm_name == "debayersplit":
            return expected_debayersplit()
        else:
            raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")


def expected_debayer2x2() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0000, 0.0333, 0.1000, 0.1333],
                    [0.1333, 0.1667, 0.2333, 0.2667],
                    [0.4000, 0.4333, 0.5000, 0.5333],
                    [0.5333, 0.5667, 0.6333, 0.6667],
                ],
                [
                    [0.1667, 0.2000, 0.2667, 0.3000],
                    [0.3000, 0.3333, 0.4000, 0.4333],
                    [0.5667, 0.6000, 0.6667, 0.7000],
                    [0.7000, 0.7333, 0.8000, 0.8333],
                ],
                [
                    [0.3333, 0.3667, 0.4333, 0.4667],
                    [0.4667, 0.5000, 0.5667, 0.6000],
                    [0.7333, 0.7667, 0.8333, 0.8667],
                    [0.8667, 0.9000, 0.9667, 1.0000],
                ],
            ]
        ]
    )


def expected_debayer3x3() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0000, 0.0667, 0.1333, 0.1333],
                    [0.2667, 0.3333, 0.4000, 0.4000],
                    [0.5333, 0.6000, 0.6667, 0.6667],
                    [0.5333, 0.6000, 0.6667, 0.6667],
                ],
                [
                    [0.1667, 0.0667, 0.2667, 0.2000],
                    [0.2667, 0.3333, 0.4000, 0.4333],
                    [0.5667, 0.6000, 0.6667, 0.7333],
                    [0.8000, 0.7333, 0.9333, 0.8333],
                ],
                [
                    [0.3333, 0.3333, 0.4000, 0.4667],
                    [0.3333, 0.3333, 0.4000, 0.4667],
                    [0.6000, 0.6000, 0.6667, 0.7333],
                    [0.8667, 0.8667, 0.9333, 1.0000],
                ],
            ]
        ]
    )


def expected_debayer5x5() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0000e00, 0.0000e00, 1.3333e-01, 1.3333e-01],
                    [1.8333e-01, 2.0833e-01, 3.2500e-01, 3.5000e-01],
                    [5.3333e-01, 5.5000e-01, 6.6667e-01, 7.0000e-01],
                    [7.8333e-01, 7.7500e-01, 9.2500e-01, 9.1667e-01],
                ],
                [
                    [1.4901e-08, 6.6667e-02, 1.5000e-01, 2.0000e-01],
                    [2.6667e-01, 2.5000e-01, 4.0000e-01, 4.0000e-01],
                    [6.0000e-01, 6.0000e-01, 7.5000e-01, 7.3333e-01],
                    [8.0000e-01, 8.5000e-01, 9.3333e-01, 1.0000e00],
                ],
                [
                    [8.3333e-02, 7.5000e-02, 2.2500e-01, 2.1667e-01],
                    [3.0000e-01, 3.3333e-01, 4.5000e-01, 4.6667e-01],
                    [6.5000e-01, 6.7500e-01, 7.9167e-01, 8.1667e-01],
                    [8.6667e-01, 8.6667e-01, 1.0000e00, 1.0000e00],
                ],
            ]
        ]
    )


def expected_debayersplit() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0000, 0.0333, 0.1000, 0.1333],
                    [0.1333, 0.1667, 0.2333, 0.2667],
                    [0.4000, 0.4333, 0.5000, 0.5333],
                    [0.5333, 0.5667, 0.6333, 0.6667],
                ],
                [
                    [0.1667, 0.0667, 0.2667, 0.2000],
                    [0.2667, 0.3333, 0.4000, 0.4333],
                    [0.5667, 0.6000, 0.6667, 0.7333],
                    [0.8000, 0.7333, 0.9333, 0.8333],
                ],
                [
                    [0.3333, 0.3667, 0.4333, 0.4667],
                    [0.4667, 0.5000, 0.5667, 0.6000],
                    [0.7333, 0.7667, 0.8333, 0.8667],
                    [0.8667, 0.9000, 0.9667, 1.0000],
                ],
            ]
        ]
    )
