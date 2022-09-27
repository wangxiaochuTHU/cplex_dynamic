//! This crate is modified based on `rplex`.
//!
//! It is a binding of IBM Cplex using its dynamic/shared library.
//!
//! Usage: After installing Cplex on your computer, copy `\IBM\ILOG\CPLEX_Studio_Community221\cplex\bin\x64_win64\cplex2210.dll` file into your crate root folder (or its subfolder).
//!
//! For example,
//!
//! [your-crate]
//! --[src]
//! ----main.rs
//! --[cplex]
//! ----cplex2210.dll
//! --Cargo.toml
//!  

#[macro_use]
extern crate dlopen_derive;
#[macro_use]
extern crate lazy_static;

mod load;

/// export all public structs and enums
pub use load::*;
