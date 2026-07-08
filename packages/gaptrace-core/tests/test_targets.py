import pytest
from gaptrace_core.targets import parse_target_id


class TestValid:
    def test_basic(self):
        assert parse_target_id("s4r3") == (4, 3)

    def test_multi_digit(self):
        assert parse_target_id("s12r345") == (12, 345)

    def test_case_insensitive(self):
        assert parse_target_id("S4R3") == (4, 3)
        assert parse_target_id("s4R3") == (4, 3)


class TestInvalid:
    @pytest.mark.parametrize(
        "bad",
        ["s4", "r3", "4r3", "s4r", "", "s-1r2", "s4r3x", "xs4r3", "s r", "s4 r3"],
    )
    def test_bad_format_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="sNrN"):
            parse_target_id(bad)

    @pytest.mark.parametrize("garbage", [None, 43, 4.3, (4, 3), ["s4r3"]])
    def test_non_string_raises_type_error(self, garbage):
        with pytest.raises(TypeError, match="sNrN"):
            parse_target_id(garbage)
