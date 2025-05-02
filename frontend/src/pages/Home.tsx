import { useState } from "react";
import { Option } from "../interface/optionInterface";
import EachQuestion from "../components/eachQuestion";
import CopyPastQuestion from "../components/copyPastQuestion";
import FileQuestion from "../components/fileQuestion";

function HomePage({ selectedOption }: { selectedOption: Option }) {


    return (<>
        <div className="h-100 flex flex-row">
            <div className="flex flex-col items-center flex-1 gap-4">
                {/* {renderOptions()} */}
            </div>
            <div className="flex-6">
                {selectedOption?.id == 1 ?
                    <EachQuestion /> :
                    selectedOption?.id == 2 ?
                        <CopyPastQuestion /> :
                        //     <EachQuestion /> :
                        selectedOption?.id == 3 ?
                            <FileQuestion /> :
                            <></>
                }
            </div>
        </div>
    </>);
}

export default HomePage;