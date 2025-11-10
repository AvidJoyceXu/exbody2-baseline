# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .g1.g1_mimic import G1Mimic
from .g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO

from .g1.g1_mimic_eval import G1MimicEval
from .g1.g1_mimic_priv_eval import G1MimicPrivEval
from .g1.g1_mimic_view_motion import G1MimicViewMotion

from .g1.g1_mimic_priv import G1MimicPriv
from .g1.g1_mimic_priv_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicPrivDistillCfgPPO
from .g1.g1_mimic_priv_distill import G1MimicPrivDistill

from .x2.x2_mimic_priv import X2MimicPriv
from .x2.x2_mimic_priv_config import X2MimicPrivCfg, X2MimicPrivCfgPPO, X2MimicPrivDistillCfgPPO
from .x2.x2_mimic_priv_distill import X2MimicPrivDistill

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1_mimic", G1Mimic, G1MimicCfg(), G1MimicCfgPPO() )
task_registry.register( "g1_mimic_eval", G1MimicEval, G1MimicCfg(), G1MimicCfgPPO() )
task_registry.register( "g1_view", G1MimicViewMotion, G1MimicCfg(), G1MimicCfgPPO() )

task_registry.register( "g1_mimic_priv", G1MimicPriv, G1MimicPrivCfg(), G1MimicPrivCfgPPO() )
task_registry.register( "g1_mimic_priv_eval", G1MimicPrivEval, G1MimicPrivCfg(), G1MimicPrivCfgPPO() )
task_registry.register( "g1_mimic_priv_distill", G1MimicPrivDistill, G1MimicPrivCfg(), G1MimicPrivDistillCfgPPO() )

task_registry.register( "x2_mimic_priv", X2MimicPriv, X2MimicPrivCfg(), X2MimicPrivCfgPPO() )
task_registry.register( "x2_mimic_priv_distill", X2MimicPrivDistill, X2MimicPrivCfg(), X2MimicPrivDistillCfgPPO() )

