# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import unittest

from omni.isaac.orbit.app import AppLauncher, run_tests


class TestAppLauncher(unittest.TestCase):
    """Test launching of the simulation app using AppLauncher."""

    def test_livestream_launch_with_env_var(self):
        """Test launching with no-keyword args but environment variables."""
        # manually set the settings as well to make sure they are set correctly
        os.environ["LIVESTREAM"] = "1"
        # everything defaults to None
        app = AppLauncher().app

        # import settings
        import carb

        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # check settings
        # -- no-gui mode
        self.assertEqual(carb_settings_iface.get("/app/window/enabled"), False)
        # -- livestream
        self.assertEqual(carb_settings_iface.get("/app/livestream/enabled"), True)

        # close the app on exit
        app.close()


if __name__ == "__main__":
    run_tests()
