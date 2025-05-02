import { Route, Routes } from "react-router";
import HomePage from "../pages/Home";
import HomeLayout from "../layout/homeLayout";
import HisToryPage from "../pages/History";

function AppRoutes() {
    return (<>
        <Routes>
            <Route path="/" element={<HomeLayout />} />
            <Route path="/history" element={<HisToryPage />} />
            <Route path="/about" element={<h1>About</h1>} />
        </Routes>
    </>);
}

export default AppRoutes;