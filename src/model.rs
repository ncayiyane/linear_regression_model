use burn::module::Module;
use burn::tensor::{backend::NdArray, Tensor};

#[derive(Module, Debug)]
pub struct LinearRegression {
    pub weight: Tensor<NdArray, 1>,
    pub bias: Tensor<NdArray, 1>,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            weight: Tensor::from_floats([[2.0]]), // Initialize with 2
            bias: Tensor::from_floats([[1.0]]),   // Initialize with 1
        }
    }

    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        x.mul(self.weight.clone()).add(self.bias.clone())
    }
}
