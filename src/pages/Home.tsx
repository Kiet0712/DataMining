import { useState } from "react";
import { Option } from "../interface/optionInterface";
import EachQuestion from "../components/eachQuestion";

function HomePage({ selectedOption }: { selectedOption: Option }) {


    return (<>
        <div className="h-100 flex flex-row">
            <div className="flex flex-col items-center flex-1 gap-4">
                {/* {renderOptions()} */}
            </div>
            <div className="flex-6">
                {selectedOption?.id == 1 ?
                    <EachQuestion /> :
                    // selectedOption?.id == 2 ?
                    //     <EachQuestion /> :
                    //     selectedOption?.id == 3 ?
                    //         <EachQuestion /> :
                    <></>
                }
            </div>
        </div>
    </>);
}

export default HomePage;