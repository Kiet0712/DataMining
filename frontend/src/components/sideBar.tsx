import { Option } from "../interface/optionInterface";

function HomeSideBar({ options, selectedOption, updateSelectedOption }: { options: Option[], selectedOption: Option, updateSelectedOption: (option: Option) => void }) {
    const renderOptions = () => {
        return options.map((option) => {
            return <div key={option.id + 'option'} className="flex flex-row items-center gap-2 h-full w-full mt-2 py-2 px-2">
                <input checked={option.id == selectedOption.id} onChange={() => updateSelectedOption(option)} type="radio" value={option.value} name="answer" id={"option" + option.id.toString()} className="cursor-pointer" />
                <label htmlFor={"option" + option.id.toString()} className="text-center cursor-pointer">{option.name}</label>
            </div>
        });
    }
    return (<>
        <div className="fixed px-3">
            {renderOptions()}
        </div>
    </>);
}

export default HomeSideBar;