//! This crate is modified based on `rplex`.
//! It is a binding of IBM Cplex using its dynamic/shared library.

#[macro_use]
extern crate dlopen_derive;
#[macro_use]
extern crate lazy_static;

mod load;

/// export all public structs and enums
pub use load::*;
