// import { initializeApp } from "firebase/app";
// import { getAuth, GoogleAuthProvider } from "firebase/auth";
// import { getFirestore } from "firebase/firestore";

// const firebaseConfig = {
//     apiKey: "AIzaSyBym4l2GY8iqF7gvxAdFGUWR56FrjIuiW0",
//     authDomain: "business-f60db.firebaseapp.com",
//     projectId: "business-f60db",
//     storageBucket: "business-f60db.appspot.com", // Fixed storage bucket URL
//     messagingSenderId: "484904888129",
//     appId: "1:484904888129:web:e223972a4d8b34d59ca3c0",
//     measurementId: "G-JW2NJYN94C",
// };

// const app = initializeApp(firebaseConfig);

// export const auth = getAuth(app);
// export const db = getFirestore(app);
// export const googleProvider = new GoogleAuthProvider();
// Firebase initialization
import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyBym4l2GY8iqF7gvxAdFGUWR56FrjIuiW0",
  authDomain: "business-f60db.firebaseapp.com",
  projectId: "business-f60db",
  storageBucket: "business-f60db.appspot.com",
  messagingSenderId: "484904888129",
  appId: "1:484904888129:web:e223972a4d8b34d59ca3c0",
};

const app = getApps().length === 0
  ? initializeApp(firebaseConfig)
  : getApps()[0];

export const auth = getAuth(app);
export const db = getFirestore(app);

