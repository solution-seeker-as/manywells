// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! Fluid properties (specs/model/pvt/): the phase correlations, and the fluid that the rest of the core calls.
//! Only the options of the v1.0.0 configuration so far: an ideal gas, dead oil and a constant liquid density.

pub mod fluid;
pub mod gas;
pub mod mixture;
pub mod oil;
