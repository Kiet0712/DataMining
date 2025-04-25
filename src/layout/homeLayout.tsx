import { useState } from "react";
import HomeHeader from "../components/header";
import HomeSideBar from "../components/sideBar";
import { Option } from "../interface/optionInterface";
import HomePage from "../pages/Home";

function HomeLayout({ children }: any) {
    const [options, setOptions] = useState<Option[]>([
        { id: 1, name: "Option 1", value: "Option 1" },
        { id: 2, name: "Option 2", value: "Option 2" },
        { id: 3, name: "Option 3", value: "Option 3" },
    ]);
    const [selectedOption, setSelectedOption] = useState<Option>(options[0]);
    const updateSelectedOption = (option: Option) => {
        setSelectedOption(option);
    }
    return (<>
        <div className="flex">
            <HomeHeader />
            <div className="flex flex-col w-full h-full flex-1 mt-20 pt-2" style={{ flex: 1 }}>
                <HomeSideBar options={options} selectedOption={selectedOption} updateSelectedOption={updateSelectedOption} />
                <div >
                    <HomePage selectedOption={selectedOption} />
                </div>
            </div>
        </div>

    </>);
}

export default HomeLayout;