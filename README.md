# Linear Regression Model in Rust

## **1. Overview**

This project implements a simple **Linear Regression Model** using the **Rust programming language** and the **Burn Library**. The model learns the function:

\(y = 2x + 1\)

with some random noise added to simulate real-world data.

## **2. Prerequisites**

Before proceeding, ensure you have the following installed:

- **Rust** (Installation: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))
- **Visual Studio Code** (Download: [https://code.visualstudio.com/](https://code.visualstudio.com/))
- **Git** (Download: [https://git-scm.com/](https://git-scm.com/))
- **Rust Analyzer Extension for VS Code**
  - Open VS Code
  - Press `Ctrl + Shift + X` to open extensions
  - Search for **Rust Analyzer** and install it

## **3. Project Setup**

### **3.1 Create a New Rust Project**

Open **VS Code Terminal** and run:

```sh
cargo new linear_regression_model
cd linear_regression_model
```

This creates a Rust project in the `linear_regression_model` folder.

### **3.2 Modify Cargo.toml**

Open `Cargo.toml` and replace its contents with:

```toml
[dependencies]
burn = { version = "0.16.0", features = ["wgpu", "train"] }
burn-ndarray = "0.16.0"
rand = "0.9.0"
rgb = "0.8.50"
textplots = "0.8.6"
```

📌 **Do not modify these dependencies** as per the assignment guidelines.

---

## **4. Implementing the Model**

### **4.1 Generate Synthetic Data**

Create `src/data.rs` and add:

```rust
use rand::Rng;

pub fn generate_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for _ in 0..n {
        let x = rng.gen_range(0.0..10.0);
        let noise: f32 = rng.gen_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise; // y = 2x + 1 + noise

        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}
```

### **4.2 Define the Model**

Create `src/model.rs`:

```rust
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
            weight: Tensor::from_floats([[2.0]]),
            bias: Tensor::from_floats([[1.0]]),
        }
    }

    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        x.mul(self.weight.clone()).add(self.bias.clone())
    }
}
```

### **4.3 Train the Model**

Create `src/train.rs`:

```rust
use burn::tensor::{backend::NdArray, Tensor};
use burn::optim::{Adam, Optimizer};
use burn::nn::loss::mse_loss;
use crate::model::LinearRegression;
use crate::data::generate_data;

pub fn train_model() {
    let model = LinearRegression::new();
    let optimizer = Adam::new(0.01);

    let (x_train, y_train) = generate_data(100);
    
    let x_train_tensor = Tensor::<NdArray, 1>::from_floats(x_train.clone());
    let y_train_tensor = Tensor::<NdArray, 1>::from_floats(y_train.clone());

    for epoch in 0..100 {
        let predictions = model.forward(x_train_tensor.clone());
        let loss = mse_loss(predictions, y_train_tensor.clone());

        optimizer.backward_step(&loss);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        }
    }
}
```

### **4.4 Evaluate the Model**

Modify `src/main.rs`:

```rust
mod model;
mod data;
mod train;

use textplots::{Chart, Plot, Shape};
use crate::train::train_model;
use crate::data::generate_data;

fn main() {
    train_model();

    let (x_test, y_test) = generate_data(10);
    let mut points = vec![];

    for i in 0..x_test.len() {
        points.push((x_test[i], y_test[i]));
    }

    println!("Model Predictions:");
    Chart::new(100, 30, 0.0, 10.0)
        .lineplot(&Shape::Lines(&points))
        .display();
}
```

---

## **5. Running the Model**

Run the program using:

```sh
cargo run
```

Expected output:

- Loss decreases over epochs
- Text-based plot of `y = 2x + 1`


