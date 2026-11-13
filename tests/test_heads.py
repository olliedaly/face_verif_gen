"""T4 and T5: the heads' logits and starting losses, and our cross-entropy.

T4, ArcFace's logits on a 2-d, 3-class toy. ArcFace's maths is easy to get subtly wrong (the
margin in the wrong column, degrees for radians, a missing clamp or fallback). In 2-d every class
and embedding can be placed at a chosen angle, so every expected logit can be written down exactly.

T5, the starting loss. Before any training, each head's loss follows from how its weights start,
so it can be predicted. A different one means a wrong s, a missing normalise or a softmax applied
twice, found before a GPU-hour is spent. And our cross-entropy must match PyTorch's, loss and
gradients, including for logits big enough to overflow e^z.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from face_verif_gen.heads import ArcFaceHead, SoftmaxHead, cross_entropy_loss

S, M = 64.0, 0.5  # ArcFace's scale and margin; M is in radians (about 28.6 degrees)
CLASS_ANGLES = [0, 90, 180]  # where the toy's three classes point, in degrees
B, C, D = 2048, 9072, 512  # batch, training identities, embedding size


def unit(degrees: float) -> torch.Tensor:
    """A batch of one 2-d embedding, shape (1, 2): the unit vector at that angle."""
    angle = math.radians(degrees)
    return torch.tensor([[math.cos(angle), math.sin(angle)]])


@pytest.fixture
def arcface():
    """ArcFace whose three classes point at 0, 90 and 180 degrees, instead of random ones."""
    head = ArcFaceHead(embed_dim=2, num_classes=3, s=S, m=M)
    # fc.weight is (num_classes, embed_dim): row j is class j's direction. Writing into a
    # trainable Parameter in place is refused while autograd is recording, hence no_grad.
    with torch.no_grad():
        head.fc.weight.copy_(torch.cat([unit(a) for a in CLASS_ANGLES]))  # (3, 2)
    return head


def test_margin_goes_on_each_rows_true_class_only(arcface):
    # One embedding at 30 degrees, three times, each copy with a different true class. In one
    # batch, so a margin in the wrong column, or every row given row 0's, fails.
    emb = unit(30).repeat(3, 1)  # (3, 2)
    labels = torch.tensor([0, 1, 2])
    # From 30 degrees to the classes at 0, 90 and 180: these angles, in every row.
    thetas = [math.radians(a) for a in (30, 60, 150)]
    # The true class gets s * cos(theta + m); the others s * cos(theta).
    expected = torch.tensor(
        [
            [S * math.cos(theta + (M if j == label else 0.0)) for j, theta in enumerate(thetas)]
            for label in labels.tolist()
        ]
    )
    torch.testing.assert_close(arcface(emb, labels), expected)


def test_lengths_are_ignored(arcface):
    # ArcFace compares directions only, so it must normalise both the embedding and the class
    # weights. The fixture's are already unit length, so make both longer and expect no change.
    labels = torch.tensor([0])
    before = arcface(unit(30), labels)
    with torch.no_grad():
        arcface.fc.weight.mul_(torch.tensor([[2.0], [3.0], [4.0]]))  # a different length per class
    torch.testing.assert_close(arcface(5 * unit(30), labels), before)


@pytest.mark.parametrize("degrees", [0, 180], ids=["on-its-class", "opposite-its-class"])
def test_gradients_stay_finite_at_cos_plus_or_minus_1(arcface, degrees):
    # acos's slope is -1 / sqrt(1 - cos^2): infinite at cos = 1 and -1, which a face lying exactly
    # on (or opposite) its class gives. The clamp keeps acos off those two points.
    emb = unit(degrees).requires_grad_()
    arcface(emb, torch.tensor([0])).sum().backward()  # sum: one number to backpropagate from
    assert torch.isfinite(emb.grad).all()
    assert torch.isfinite(arcface.fc.weight.grad).all()


def test_past_pi_minus_m_the_fallback_is_used(arcface):
    # 170 degrees from its class: 170 + 28.6 is past 180, where cos turns back up, so
    # cos(theta + m) would reward a face for being further away. The head must use the fallback.
    theta = math.radians(170)
    logits = arcface(unit(170), torch.tensor([0]))
    fallback = S * (math.cos(theta) - M * math.sin(M))  # -78.4; cos(theta + m) would give -60.7
    torch.testing.assert_close(logits[0, 0], torch.tensor(fallback))


# Both starting losses: loss = log-sum-exp of the logits - the true logit. If every logit were 0,
# the log-sum-exp would be ln C. Logits spread with variance sigma^2 raise it to ln C + sigma^2 / 2,
# because the average of e^z for z ~ N(0, sigma^2) is e^(sigma^2 / 2).
# Tolerances come from 100 seeds at B = 2048: softmax within 0.04 of its formula (sd 0.013),
# ArcFace within 0.2 (about 0.1 low on average, sd 0.06). Both are 7 sd or more, so any seed
# passes; the seed is only there so that a failure repeats.


def test_softmax_head_starting_loss():
    """About ln C + 1/6 = 9.28.

    A logit sums D products w * x. nn.Linear starts each w uniform on (-1/sqrt(D), 1/sqrt(D)),
    variance 1/(3D); each x is N(0, 1), as after IResNet's last BatchNorm. So one product has
    variance 1/(3D), and D of them sum to sigma^2 = 1/3. The true logit averages 0.
    """
    torch.manual_seed(0)
    head = SoftmaxHead(embed_dim=D, num_classes=C)
    emb = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))
    loss = cross_entropy_loss(head(emb, labels), labels)
    # Not wider: a softmax applied twice gives ln C = 9.11, only 0.17 lower.
    assert loss.item() == pytest.approx(math.log(C) + (1 / 3) / 2, abs=0.1)


def test_arcface_head_starting_loss():
    """About s * sin(m) + ln(C - 1) + 4 = 43.8.

    Random directions in D dimensions are nearly perpendicular: cos has mean 0 and variance 1/D.
    So the true class sits at about 90 + 28.6 degrees, and its logit is s * cos(90 deg + m) =
    -s * sin(m) = -30.7, too small to count in the sum. The other C - 1 logits are s * cos, with
    variance s^2 / D = 8, so their log-sum-exp is ln(C - 1) + 8 / 2.
    """
    torch.manual_seed(0)
    head = ArcFaceHead(embed_dim=D, num_classes=C, s=S, m=M)
    emb = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))
    loss = cross_entropy_loss(head(emb, labels), labels)
    sigma_squared = S**2 / D
    expected = S * math.sin(M) + math.log(C - 1) + sigma_squared / 2
    # Still far below what a wrong head gives: m = 0.4 moves it by 6, a wrong s by much more.
    assert loss.item() == pytest.approx(expected, abs=0.5)


@pytest.mark.parametrize("scale", [1, 100, 1e4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_cross_entropy_matches_pytorchs(scale, dtype):
    """Loss and gradients match F.cross_entropy, also for logits far past e^z's overflow (~88)."""
    torch.manual_seed(0)
    values = (torch.randn(8, C) * scale).to(dtype)  # 8 rows: this checks arithmetic, not averages
    labels = torch.randint(0, C, (8,))
    # Two separate leaves holding the same values, so each function gets its own .grad. The
    # reference gets them in float32: it computes in its input's dtype, ours always in float32.
    ours_in = values.clone().requires_grad_()
    ref_in = values.float().clone().requires_grad_()
    ours = cross_entropy_loss(ours_in, labels)
    ref = F.cross_entropy(ref_in, labels)
    ours.backward()
    ref.backward()
    torch.testing.assert_close(ours, ref)
    # A leaf's .grad has the leaf's dtype, bf16 when the input was, so compare in that dtype.
    torch.testing.assert_close(ours_in.grad, ref_in.grad.to(dtype))
