import React from 'react';
import './App.css';
import logo from './logo.svg';
import { toast, ToastContainer } from 'react-toastify';
import AppRoutes from './routes/routes';
import HomeLayout from './layout/homeLayout';

function App() {
  return (
    <div className="App">
      <AppRoutes />
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick={false}
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </div>
  );
}

export default App;
