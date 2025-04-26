import { Route, Routes } from "react-router";
import HomePage from "../pages/Home";
import HomeLayout from "../layout/homeLayout";

function AppRoutes() {
    return (<>
        <Routes>
            <Route path="/" element={<HomeLayout />} />
            <Route path="/about" element={<h1>About</h1>} />
        </Routes>
    </>);
}

export default AppRoutes;