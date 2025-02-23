use burn::tensor::{backend::NdArray, Tensor};
use burn::optim::{Adam, Optimizer};
use burn::nn::loss::mse_loss;
use crate::model::LinearRegression;
use crate::data::generate_data;

pub fn train_model() {
    let model = LinearRegression::new();
    let optimizer = Adam::new(0.01); // Learning rate

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
