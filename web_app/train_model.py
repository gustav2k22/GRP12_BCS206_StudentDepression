#!/usr/bin/env python3
"""
Student Depression Model Training
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class StudentDepressionModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'Depression'
        
    def load_and_preprocess_data(self):
        """Load and preprocess data exactly as in model.ipynb"""
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values if any
        print("Checking for missing values...")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        if missing_values.sum() > 0:
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != self.target_column:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove quotes from Sleep Duration if present
        if 'Sleep Duration' in df.columns:
            df['Sleep Duration'] = df['Sleep Duration'].str.replace("'", "")
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != self.target_column]
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded {col}: {le.classes_}")
        
        # Prepare features and target
        X = df.drop([self.target_column], axis=1)
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {self.feature_columns}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_stacking_model(self):
        """Create the exact stacking model from model.ipynb"""
        print("🤖 Creating Stacking Classifier...")
        
        # Define base models exactly as in the notebook
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('nb', GaussianNB())
        ]
        
        # Define meta-model
        meta_model = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,  # 5-fold cross-validation
            stack_method='predict_proba',  # Use probabilities
            n_jobs=-1
        )
        
        print("Base Models:")
        for name, model in base_models:
            print(f"  • {name}: {model.__class__.__name__}")
        print(f"Meta Model: {meta_model.__class__.__name__}")
        
        return stacking_clf
    
    def train_model(self, X, y):
        """Train the model with hyperparameter tuning"""
        print("🚀 Training Stacking Classifier...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Testing set: {X_test_scaled.shape}")
        
        # Create the stacking model
        stacking_clf = self.create_stacking_model()
        
        # Hyperparameter tuning (simplified for faster training)
        print("⚙️ Performing hyperparameter tuning...")
        
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [10, None],
            'svm__C': [1, 10],
            'gb__n_estimators': [100],
            'gb__learning_rate': [0.1],
            'final_estimator__C': [1, 10]
        }
        
        grid_search = GridSearchCV(
            stacking_clf,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        print(f"🏆 Best Parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"Best Cross-Validation AUC Score: {best_cv_score:.4f}")
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Test Set Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_model(self, output_path='model_components.pkl'):
        """Save the trained model and preprocessing components"""
        print(f"💾 Saving model components to {output_path}...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Model saved successfully!")
        return output_path
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_columns
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\n📈 Top 10 Feature Importance:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None

def main():
    """Main training function"""
    print("🚀 Starting Student Depression Model Training")
    print("=" * 60)
    
    # Paths
    data_path = "../data/raw/student_depression_dataset.csv"
    model_output_path = "model_components.pkl"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset file not found at {data_path}")
        return
    
    # Initialize trainer
    trainer = StudentDepressionModelTrainer(data_path)
    
    try:
        # Load and preprocess data
        X, y = trainer.load_and_preprocess_data()
        
        # Train model
        trainer.train_model(X, y)
        
        # Get feature importance
        trainer.get_feature_importance()
        
        # Save model
        trainer.save_model(model_output_path)
        
        print("\n" + "=" * 60)
        print("✅ Model training completed successfully!")
        print(f"📁 Model saved as: {model_output_path}")
        print("🌐 Ready to integrate with web application!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
