"""Stage curriculum schedules for masked prediction + SIGReg gating."""
from tomojepa.swinjepa.losses import lambda_schedule


def test_fine_in_ramps_s1_s2():
    bases = (1.0, 1.0, 1.0, 1.0)
    lam0 = lambda_schedule(0, 1000, 0.25, 0.1, bases, (1, 2), stage_curriculum="fine_in")
    assert lam0[0] == 0.1 and lam0[1] == 0.1
    assert lam0[2] == 1.0 and lam0[3] == 1.0
    lam_end = lambda_schedule(250, 1000, 0.25, 0.1, bases, (1, 2), stage_curriculum="fine_in")
    assert lam_end == [1.0, 1.0, 1.0, 1.0]


def test_coarse_in_starts_at_s4():
    bases = (1.0, 1.0, 1.0, 1.0)
    lam0 = lambda_schedule(0, 1000, 0.25, 0.1, bases, (1, 2),
                           stage_curriculum="coarse_in", coarse_ramp_stages=(3, 2, 1))
    assert lam0[3] == 1.0          # s4 full
    assert lam0[2] == 0.1          # s3 waiting
    assert lam0[1] == 0.1          # s2 waiting
    assert lam0[0] == 0.1          # s1 waiting


def test_coarse_in_staggered_ramps():
    bases = (1.0, 1.0, 1.0, 1.0)
    # warmup=250, 3 segments ~83 steps each
    lam_s3 = lambda_schedule(40, 1000, 0.25, 0.1, bases, (1, 2),
                               stage_curriculum="coarse_in", coarse_ramp_stages=(3, 2, 1))
    assert lam_s3[3] == 1.0
    assert lam_s3[2] > 0.1         # s3 ramping
    assert lam_s3[1] == 0.1
    assert lam_s3[0] == 0.1

    lam_all = lambda_schedule(250, 1000, 0.25, 0.1, bases, (1, 2),
                              stage_curriculum="coarse_in", coarse_ramp_stages=(3, 2, 1))
    assert lam_all == [1.0, 1.0, 1.0, 1.0]
